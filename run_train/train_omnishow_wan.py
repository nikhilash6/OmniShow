import argparse
import faulthandler
import os
import traceback
import warnings

import accelerate
import torch
from PIL import Image
from torch.distributed.elastic.multiprocessing.errors import record

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadVideo,
    RouteByType,
    SequencialProcess,
    ToAbsolutePath,
)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, max_num_omnishow_ref_images
from diffsynth.diffusion import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"



CONDITION_ALIASES = {
    "t2v": ("text",),
    "r2v": ("text", "ref"),
    "a2v": ("text", "firstframe", "audio"),
    "ra2v": ("text", "ref", "audio"),
    "rp2v": ("text", "ref", "pose"),
    "rap2v": ("text", "ref", "audio", "pose"),
}

VALID_ATOMS = {"text", "ref", "pose", "audio", "firstframe"}


def parse_conditions(spec: str) -> set:
    """Accept either alias (t2v/r2v/...) or comma-joined atoms (text,ref,audio)."""
    spec = (spec or "text").strip().lower()
    if spec in CONDITION_ALIASES:
        atoms = set(CONDITION_ALIASES[spec])
    else:
        atoms = {x.strip() for x in spec.split(",") if x.strip()}
    unknown = atoms - VALID_ATOMS
    if unknown:
        raise ValueError(
            f"Unknown condition atoms: {unknown}. "
            f"Valid atoms: {sorted(VALID_ATOMS)} or aliases {sorted(CONDITION_ALIASES)}"
        )
    atoms.add("text")
    return atoms


def _letterbox_to_white(image: Image.Image, width: int, height: int) -> Image.Image:
    """Keep aspect ratio and paste into a white canvas of (width,height)."""
    canvas = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        return canvas
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    resized = image.resize((dst_w, dst_h), resample=resample)
    left = (width - dst_w) // 2
    top = (height - dst_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _normalize_ref_images(ref_images, width: int, height: int):
    if ref_images is None:
        return None
    if not isinstance(ref_images, list):
        ref_images = [ref_images]
    out = []
    for img in ref_images:
        if img is None:
            continue
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            img = _letterbox_to_white(img.convert("RGB"), int(width), int(height))
        out.append(img)
    return out if out else None



class WanOmniShowTrainingModule(DiffusionTrainingModule):
    """Stripped-down WanTrainingModule that builds inputs from CSV columns
    directly without precomputed latent inputs."""

    def __init__(
        self,
        conditions,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. The framework will force-enable it."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        default_tokenizer_path = os.path.join(
            os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"),
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
        )
        tokenizer_config = (
            ModelConfig(path=default_tokenizer_path)
            if tokenizer_path is None and os.path.isdir(default_tokenizer_path) else
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None else ModelConfig(tokenizer_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            omnishow_audio_encoder=True,
            redirect_common_files=False,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        self.conditions = conditions
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def _collect_ref_images(self, data):
        images = []
        for key in ("ref_image_human", "ref_image_object"):
            val = data.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                images.extend(val)
            else:
                images.append(val)
        return images if images else None

    def get_pipeline_inputs(self, data):
        # Convert one dataset row into WanVideoPipeline training inputs.
        conditions = self.conditions
        prompt = data.get("text_prompt", "") or ""
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}

        if "target_video" not in data:
            raise ValueError("train set must contain 'target_video' column.")
        video = data["target_video"]
        inputs_shared = {
            "input_video": video,
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
        }

        inputs_shared["is_omnishow_enable_ref"] = ("ref" in conditions)

        if "ref" in conditions:
            ref_images = self._collect_ref_images(data)
            ref_images = _normalize_ref_images(ref_images, inputs_shared["width"], inputs_shared["height"])
            if ref_images is not None:
                inputs_shared["ref_images"] = ref_images

            base = (ref_images or [])[:max_num_omnishow_ref_images]
            if len(base) < max_num_omnishow_ref_images:
                w, h = video[0].size
                base = base + [Image.new("RGB", (w, h), (0, 0, 0)) for _ in range(max_num_omnishow_ref_images - len(base))]
            inputs_shared["omnishow_reference_image"] = base

        if "pose" in conditions and "pose_video" in data:
            inputs_shared["pose_video"] = data["pose_video"]

        if "audio" in conditions and "audio" in data:
            audio = data["audio"]
            if isinstance(audio, tuple):
                waveform, sr = audio
            else:
                waveform, sr = audio, 16000
            inputs_shared["input_audio"] = waveform
            inputs_shared["audio_sample_rate"] = sr

        inputs_shared["is_omnishow_enable_audio"] = ("audio" in conditions)

        if "firstframe" in conditions:
            inputs_shared["input_image"] = video[0]

        inputs_shared.update({
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        })
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def optimizer_param_groups(self, learning_rate, weight_decay, args=None):
        learning_rate_audio = getattr(args, "learning_rate_audio", None) if args is not None else None
        if learning_rate_audio is None:
            return [{"params": list(self.trainable_modules()), "lr": learning_rate, "weight_decay": weight_decay}]

        audio_params = []
        other_params = []
        audio_param_names = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_audio_param = (
                name.startswith("pipe.dit.omnishow_audio_projector.")
                or ".glca." in name
            )
            if is_audio_param:
                audio_params.append(param)
                audio_param_names.append(name)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({
                "params": other_params,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "name": "non_audio",
            })
        if audio_params:
            groups.append({
                "params": audio_params,
                "lr": float(learning_rate_audio),
                "weight_decay": float(weight_decay),
                "name": "omnishow_audio",
            })

        print(
            "[train_omnishow_wan] optimizer param groups: "
            f"non_audio_params={len(other_params)} lr={float(learning_rate):.3e}, "
            f"audio_params={len(audio_params)} lr={float(learning_rate_audio):.3e}"
        )
        if audio_param_names:
            print("[train_omnishow_wan] audio lr param examples:")
            for n in audio_param_names[:12]:
                print(f"  - {n}")
            if len(audio_param_names) > 12:
                print(f"  ... {len(audio_param_names) - 12} more")
        if not audio_params:
            print("[train_omnishow_wan] WARNING: --learning_rate_audio was set but no OmniShow audio params were found.")
        return groups



def wan_omnishow_parser():
    parser = argparse.ArgumentParser(description="OmniShow Wan2.1-I2V training (text/ref/pose/audio).")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--gen_task", type=str, default="text",
        help="Generation task alias (t2v/r2v/a2v/ra2v/rp2v/rap2v) or comma list (text,ref,audio,pose).",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--learning_rate_audio",
        type=float,
        default=None,
        help="Optional LR for OmniShow audio params: dit.omnishow_audio_projector and all GLCA modules.",
    )
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    return parser


@record
def main():
    faulthandler.enable(all_threads=True)
    parser = wan_omnishow_parser()
    try:
        args = parser.parse_args()
        conditions = parse_conditions(args.gen_task)
        print(f"[train_omnishow_wan] gen_task={args.gen_task} atoms={sorted(conditions)}")

        if "audio" in conditions:
            # Force OmniShow audio-compatible DiT construction for audio training.
            force_target = None
            if getattr(args, "model_paths", None):
                try:
                    import json as _json
                    mp = _json.loads(args.model_paths)
                    if isinstance(mp, list):
                        for p in mp:
                            if isinstance(p, str) and ("diffusion_pytorch_model" in p) and p.endswith(".safetensors"):
                                force_target = p
                                break
                        if force_target is None:
                            for p in mp:
                                if isinstance(p, str) and p.endswith(".safetensors"):
                                    force_target = p
                                    break
                except Exception:
                    force_target = None
            if force_target is None and getattr(args, "model_id_with_origin_paths", None):
                s = str(args.model_id_with_origin_paths)
                for item in [x.strip() for x in s.split(",") if x.strip()]:
                    if ":" not in item:
                        continue
                    model_id, origin = item.split(":", 1)
                    if "diffusion_pytorch_model" in origin:
                        base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")
                        cand_dir = os.path.join(base, model_id)
                        if os.path.isdir(cand_dir):
                            force_target = cand_dir
                            break
                    if origin.endswith(".safetensors"):
                        if os.path.exists(origin):
                            force_target = origin
                            break
                        base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")
                        cand = os.path.join(base, model_id, origin)
                        if os.path.exists(cand):
                            force_target = cand
                            break
            if force_target is not None:
                os.environ["DIFFSYNTH_OMNISHOW_FORCE_DIT"] = str(force_target)
                os.environ["DIFFSYNTH_LOAD_STATE_DICT_STRICT"] = "false"
                hint_src = " ".join([
                    str(getattr(args, "model_id_with_origin_paths", "") or ""),
                    str(force_target or ""),
                ]).lower()
                if "i2v" in hint_src and "14b" in hint_src and "720" in hint_src:
                    variant = "i2v_14b_720"
                elif "i2v" in hint_src and "14b" in hint_src and "480" in hint_src:
                    variant = "i2v_14b_480"
                else:
                    raise ValueError(
                        "OmniShow audio training only supports Wan2.1-I2V-14B-480P / 720P. "
                        f"Cannot infer variant from: {hint_src!r}"
                    )
                os.environ["DIFFSYNTH_OMNISHOW_FORCE_DIT_VARIANT"] = variant
                print(f"[train_omnishow_wan] OmniShow audio: force_dit={force_target}, variant={variant}")

        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        )

        video_operator = UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        image_operator = UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
        _ = RouteByType(operator_map=[
            (str, ToAbsolutePath(args.dataset_base_path)),
            (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path))),
        ])

        data_file_keys = ["target_video"]
        if "ref" in conditions:
            data_file_keys += ["ref_image_human", "ref_image_object"]
        if "pose" in conditions:
            data_file_keys.append("pose_video")
        if "audio" in conditions:
            data_file_keys.append("audio")

        special_operator_map = {
            "target_video": video_operator,
            "ref_image_human": image_operator,
            "ref_image_object": image_operator,
            "pose_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                args.num_frames, 4, 1,
                frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16),
            ),
            "audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=data_file_keys,
            main_data_operator=video_operator,
            special_operator_map=special_operator_map,
        )

        model = WanOmniShowTrainingModule(
            conditions=conditions,
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            tokenizer_path=args.tokenizer_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            preset_lora_path=args.preset_lora_path,
            preset_lora_model=args.preset_lora_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            task=args.task,
            device="cpu" if args.initialize_model_on_cpu else accelerator.device,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
        model_logger = ModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
        accelerator.print(
            f"[train_omnishow_wan] dataset_size={len(dataset)}, dataset_repeat={args.dataset_repeat}, "
            f"task={args.task}, output_path={args.output_path}"
        )
        accelerator.print(
            f"[train_omnishow_wan] save_steps={args.save_steps}, num_epochs={args.num_epochs}, "
            f"resolution={args.width}x{args.height}, num_frames={args.num_frames}"
        )
        launcher_map = {
            "sft:data_process": launch_data_process_task,
            "sft": launch_training_task,
            "sft:train": launch_training_task,
        }
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
