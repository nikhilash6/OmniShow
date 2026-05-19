import argparse
import csv
import os

import torch
from PIL import Image

from diffsynth.core.data.operators import ImageCropAndResize, LoadVideo
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video, save_video_with_audio


DEFAULT_NEGATIVE_PROMPT = (
    "overly saturated, overexposed, static, blurry details, subtitles, style, artwork, painting, "
    "still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, motionless frames, cluttered background, three legs, crowded background, walking backwards"
)


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


def _is_empty(x) -> bool:
    return x is None or (isinstance(x, str) and x.strip() == "")


def resolve_path(base_path: str, maybe_rel: str) -> str:
    if _is_empty(maybe_rel):
        return ""
    p = str(maybe_rel)
    return os.path.join(base_path, p) if base_path else p


def load_csv_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_model_configs(model_id: str, dit_checkpoint: str | None = None):
    # Build model configs and optionally replace DiT with a local checkpoint.
    if dit_checkpoint is not None and str(dit_checkpoint).strip() != "":
        dit_cfg = ModelConfig(path=str(dit_checkpoint))
    else:
        dit_cfg = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    return [
        dit_cfg,
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]


def load_ref_images(row: dict, base_path: str, height: int, width: int):
    def _letterbox_to_white(image: Image.Image, width: int, height: int) -> Image.Image:
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

    images = []
    for key in ("ref_image_human", "ref_image_object"):
        val = row.get(key)
        if _is_empty(val):
            continue
        full_path = resolve_path(base_path, val)
        img = Image.open(full_path).convert("RGB")
        img = _letterbox_to_white(img, width, height)
        images.append(img)
    if not images:
        raise ValueError(
            "Missing ref images for 'ref' condition: at least one of "
            "'ref_image_human' / 'ref_image_object' must be non-empty in the CSV row."
        )
    return images


def load_pose_video(row: dict, base_path: str, num_frames: int, height: int, width: int):
    pose_video_path = row.get("pose_video")
    if _is_empty(pose_video_path):
        raise ValueError("Missing 'pose_video' in CSV row for 'pose' condition.")
    full_path = resolve_path(base_path, pose_video_path)
    loader = LoadVideo(
        num_frames,
        4,
        1,
        frame_processor=ImageCropAndResize(height, width, None, 16, 16),
    )
    return loader(full_path)


def load_audio(row: dict, base_path: str, target_sr: int = 16000):
    audio_path = row.get("audio")
    if _is_empty(audio_path):
        raise ValueError("Missing 'audio' in CSV row for 'audio' condition.")
    full_path = resolve_path(base_path, audio_path)
    import librosa

    waveform, sr = librosa.load(full_path, sr=target_sr, mono=True)
    return waveform, sr, full_path


def load_first_frame(row: dict, base_path: str, height: int, width: int):
    input_image = row.get("input_image")
    if _is_empty(input_image):
        raise ValueError("Missing 'input_image' in CSV row for 'firstframe' condition.")
    full_path = resolve_path(base_path, input_image)
    img = Image.open(full_path).convert("RGB")
    img = ImageCropAndResize(height, width, None, 16, 16)(img)
    return img


def run_single_csv(
    pipe: WanVideoPipeline,
    csv_path: str,
    output_dir: str,
    base_path: str,
    conditions: set,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    fps: int,
):
    # Map each CSV row to pipeline inputs and save one video per row.
    rows = load_csv_rows(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in enumerate(rows):
        prompt = row.get("text_prompt", "") or ""
        negative_prompt = row.get("negative_prompt")
        negative_prompt = negative_prompt if not _is_empty(negative_prompt) else DEFAULT_NEGATIVE_PROMPT

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "seed": int(row.get("seed")) if not _is_empty(row.get("seed")) else seed,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "is_omnishow_enable_audio": ("audio" in conditions),
            "is_omnishow_enable_ref": ("ref" in conditions),
        }

        audio_path = None
        if "ref" in conditions:
            call_kwargs["ref_images"] = load_ref_images(row, base_path, height, width)
        if "pose" in conditions:
            call_kwargs["pose_video"] = load_pose_video(row, base_path, num_frames, height, width)
        if "audio" in conditions:
            waveform, sr, audio_path = load_audio(row, base_path, target_sr=16000)
            call_kwargs["input_audio"] = waveform
            call_kwargs["audio_sample_rate"] = sr
        if "firstframe" in conditions:
            call_kwargs["input_image"] = load_first_frame(row, base_path, height, width)

        output_name = row.get("output_name") or f"sample_{idx:04d}"
        video = pipe(**call_kwargs)
        out_path = os.path.join(output_dir, f"{output_name}.mp4")
        if "audio" in conditions and audio_path is not None:
            save_video_with_audio(video, out_path, audio_path=audio_path, fps=fps, quality=5)
        else:
            save_video(video, out_path, fps=fps, quality=5)


def main():
    parser = argparse.ArgumentParser(description="Wan2.1-I2V OmniShow inference (text/ref/pose/audio).")
    parser.add_argument("--csv", type=str, required=True, help="Path to a single CSV file.")
    parser.add_argument("--output_dir", type=str, default="outputs/omnishow", help="Output directory.")
    parser.add_argument("--base_path", type=str, default="", help="Base path for relative paths in the CSV.")
    parser.add_argument(
        "--gen_task",
        type=str,
        default="text",
        help="Generation task alias (t2v/r2v/a2v/ra2v/rp2v/rap2v) or comma list (text,ref,audio,pose,firstframe).",
    )
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P")
    parser.add_argument(
        "--dit_checkpoint",
        type=str,
        default=None,
        help="Optional local DiT checkpoint file to replace the base DiT weights (e.g., outputs/train_.../epoch-xx.safetensors).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_stride", type=int, default=192)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    conditions = parse_conditions(args.gen_task)
    print(f"[infer_omnishow_wan] gen_task={args.gen_task} atoms={sorted(conditions)}")

    if "audio" in conditions:
        # Force OmniShow audio-compatible DiT construction for audio tasks.
        if args.dit_checkpoint is not None and str(args.dit_checkpoint).strip() != "":
            os.environ["DIFFSYNTH_OMNISHOW_FORCE_DIT"] = str(args.dit_checkpoint)
        else:
            base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")
            os.environ["DIFFSYNTH_OMNISHOW_FORCE_DIT"] = os.path.join(base, args.model_id)
        mid = str(args.model_id).lower()
        if "i2v" in mid and "14b" in mid and "720" in mid:
            variant = "i2v_14b_720"
        elif "i2v" in mid and "14b" in mid and "480" in mid:
            variant = "i2v_14b_480"
        else:
            raise ValueError(
                "OmniShow audio inference only supports Wan2.1-I2V-14B-480P / 720P. "
                f"Got model_id={args.model_id!r}"
            )
        os.environ["DIFFSYNTH_OMNISHOW_FORCE_DIT_VARIANT"] = variant

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=build_model_configs(args.model_id, args.dit_checkpoint),
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        omnishow_audio_encoder=True,
        redirect_common_files=False,
    )

    run_single_csv(
        pipe=pipe,
        csv_path=args.csv,
        output_dir=args.output_dir,
        base_path=args.base_path,
        conditions=conditions,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=args.tiled,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
