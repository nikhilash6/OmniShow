import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniShowAudioEncoder(nn.Module):
    """OmniShow audio encoder (Wav2Vec2 + context packing).

    Contract:
      - Input waveform: (B, T_samples) @ 16kHz, or (T_samples,)
      - Output: audio_context (B, T_latent, w, packed_dim)
        where w=5 and packed_dim = 13 * 768 = 9984 (out.hidden_states for wav2vec2-base).
    """

    # Fixed audio rate and video fps used for audio-to-latent alignment.
    sample_rate_hz: int = 16000
    video_fps: int = 16

    def __init__(
        self,
        wav2vec_model_name_or_path: str | None = None,
        out_dim: int | None = None,
        window_size: int = 5,
        stride: int = 4,
        video_fps: int | None = None,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.video_fps = int(video_fps) if video_fps is not None else int(type(self).video_fps)

        # Resolve Wav2Vec2 weights from argument, local model base, or HuggingFace id.
        model_name_or_path = str(wav2vec_model_name_or_path).strip() if wav2vec_model_name_or_path is not None else ""

        if not model_name_or_path:
            base = os.getenv("DIFFSYNTH_MODEL_BASE_PATH", "").strip()
            if base:
                candidates = [
                    os.path.join(base, "facebook", "wav2vec2-base-960h"),
                    os.path.join(base, "wav2vec2-base-960h"),
                    os.path.join(base, "wav2vec2"),
                ]
                for c in candidates:
                    if os.path.isdir(c):
                        model_name_or_path = c
                        break

        if not model_name_or_path:
            model_name_or_path = "facebook/wav2vec2-base-960h"

        # Wav2Vec2 is lazy-loaded and kept outside registered submodules.
        self._wav2vec_model_name_or_path = model_name_or_path
        object.__setattr__(self, "_processor", None)
        object.__setattr__(self, "_wav2vec", None)

        # Default packed feature dimension for wav2vec2-base hidden states.
        self.packed_dim = 13 * 768
        self.out_dim = int(out_dim) if out_dim is not None else self.packed_dim

    def _ensure_loaded(self, device: torch.device):
        wav2vec = getattr(self, "_wav2vec", None)
        processor = getattr(self, "_processor", None)

        if wav2vec is not None and processor is not None:
            wav2vec.to(device=device)
            wav2vec.eval()
            return

        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(self._wav2vec_model_name_or_path)

        # Load the frozen Wav2Vec2 encoder without distributed parameter partitioning.
        _hf_ds_int = None
        _saved_ref = None
        # Remove optional weight normalization wrappers before using Conv1d features.
        try:
            from transformers.integrations import deepspeed as _hf_ds_int

            _saved_ref = getattr(_hf_ds_int, "_hf_deepspeed_config_weak_ref", None)
            _hf_ds_int._hf_deepspeed_config_weak_ref = None
        except Exception:
            _hf_ds_int = None

        try:
            wav2vec = Wav2Vec2Model.from_pretrained(self._wav2vec_model_name_or_path)
        finally:
            if _hf_ds_int is not None:
                try:
                    _hf_ds_int._hf_deepspeed_config_weak_ref = _saved_ref
                except Exception:
                    pass

        try:
            from torch.nn.utils import remove_weight_norm

            removed = 0
            for mod in wav2vec.modules():
                if isinstance(mod, nn.Conv1d):
                    has_legacy = hasattr(mod, "weight_g") and hasattr(mod, "weight_v")
                    has_param = hasattr(mod, "parametrizations") and hasattr(mod.parametrizations, "weight")
                    if has_legacy or has_param:
                        try:
                            remove_weight_norm(mod)
                            removed += 1
                        except Exception:
                            pass
            if removed:
                print(f"[OmniShowAudioEncoder] removed weight_norm from {removed} Conv1d layers in wav2vec2")
        except Exception:
            pass

        # Validate that Wav2Vec2 convolution weights are materialized correctly.
        for n, mod in wav2vec.named_modules():
            if isinstance(mod, nn.Conv1d):
                w = getattr(mod, "weight", None)
                if torch.is_tensor(w) and w.numel() == 0:
                    raise RuntimeError(
                        f"wav2vec2 Conv1d weight is not materialized (numel=0) at {n}. "
                        "This is typically caused by DeepSpeed ZeRO-3 partitioning. "
                        "Ensure wav2vec2 is constructed outside `deepspeed.zero.Init()` (this repo disables it during load)."
                    )
                if torch.is_tensor(w) and w.dim() != 3:
                    raise RuntimeError(
                        f"wav2vec2 Conv1d weight has unexpected dim={w.dim()} shape={tuple(w.shape)} at {n}. "
                        "This usually indicates weight_norm/ZeRO interaction or a corrupted checkpoint."
                    )

        wav2vec.to(device=device)
        wav2vec.eval()
        for p in wav2vec.parameters():
            p.requires_grad_(False)

        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_wav2vec", wav2vec)

    @staticmethod
    def _linear_interpolate_1d(features_btd: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linear interpolate along time: (B, T, D) -> (B, target_len, D)."""
        if features_btd.dim() != 3:
            raise ValueError(f"features must be (B,T,D), got {tuple(features_btd.shape)}")
        x = features_btd.transpose(1, 2)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=True)
        return x.transpose(1, 2)

    @staticmethod
    def _pack_sliding_window(features_btd: torch.Tensor, window_size: int) -> torch.Tensor:
        """Center sliding window with edge replication.

        Input:  (B, T, D)
        Output: (B, T, w, D) where w=window_size
        """
        if window_size % 2 != 1:
            raise ValueError(f"window_size must be odd, got {window_size}")
        B, T, D = features_btd.shape
        pad = window_size // 2

        out = features_btd.new_empty((B, T, window_size, D))
        if pad == 0:
            out[:, :, 0, :] = features_btd
            return out

        offsets = torch.arange(-pad, pad + 1, device=features_btd.device)
        for t in range(T):
            idx = (t + offsets).clamp(0, T - 1)
            out[:, t, :, :] = features_btd.index_select(1, idx)
        return out

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
        if waveform is None:
            raise ValueError("waveform must not be None")
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"waveform must have shape (B, T) or (T,), got {tuple(waveform.shape)}")

        num_latent_frames = int(num_latent_frames)
        if num_latent_frames <= 0:
            raise ValueError(f"num_latent_frames must be > 0, got {num_latent_frames}")

        # Convert latent length to the video-frame timeline.
        t_video = 4 * (num_latent_frames - 1) + 1
        expected_samples = int(round(self.sample_rate_hz * (t_video / float(self.video_fps))))

        # Match waveform duration to the requested latent length.
        if waveform.shape[1] > expected_samples:
            waveform = waveform[:, :expected_samples]
        elif waveform.shape[1] < expected_samples:
            raise ValueError(
                f"waveform too short for num_latent_frames={num_latent_frames} "
                f"(expected_samples={expected_samples}, got={int(waveform.shape[1])}, video_fps={self.video_fps}). "
                "Please pad/extend audio to match the expected duration, or adjust num_frames/video_fps."
            )

        waveform = waveform.to(dtype=torch.float32)

        self._ensure_loaded(device=waveform.device)

        processor = getattr(self, "_processor", None)
        wav2vec = getattr(self, "_wav2vec", None)
        if processor is None or wav2vec is None:
            raise RuntimeError("wav2vec2 is not initialized; _ensure_loaded() should have set it")

        inputs = processor(
            waveform,
            sampling_rate=self.sample_rate_hz,
            return_tensors="pt",
            padding=True,
        )
        # Normalize processor output to the shape expected by Wav2Vec2Model.
        input_values = inputs.input_values
        if torch.is_tensor(input_values) and input_values.dim() == 3 and input_values.shape[1] == 1:
            input_values = input_values.squeeze(1)
        input_values = input_values.to(device=waveform.device, dtype=torch.float32)
        out = wav2vec(input_values, output_hidden_states=True)
        hidden = list(out.hidden_states) if out.hidden_states is not None else []
        if not hidden:
            raise RuntimeError("wav2vec2 did not return hidden_states; please ensure output_hidden_states=True")
        if len(hidden) != 13:
            raise RuntimeError(
                f"Unexpected wav2vec2 hidden_states count={len(hidden)}; expected 13 for wav2vec2-base. "
                "If you are using a different wav2vec2 variant, adjust projector/input dims accordingly."
            )
        # Concatenate hidden states and align local audio windows to latent frames.
        feats = torch.cat(hidden, dim=-1)
        self.packed_dim = int(feats.shape[-1])
        self.out_dim = int(self.packed_dim)

        feats = self._linear_interpolate_1d(feats, target_len=t_video)

        feats = self._pack_sliding_window(feats, window_size=self.window_size)

        idx = torch.arange(0, t_video, self.stride, device=feats.device)
        if idx.numel() == 0:
            raise RuntimeError(f"invalid stride={self.stride} for t_video={t_video}")
        if idx.numel() != num_latent_frames:
            idx = idx[:num_latent_frames]
            if idx.numel() < num_latent_frames:
                last = idx[-1]
                pad_n = num_latent_frames - idx.numel()
                idx = torch.cat([idx, last.repeat(pad_n)], dim=0)
        feats = feats.index_select(1, idx)
        return feats
