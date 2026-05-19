#!/usr/bin/env bash

set -euo pipefail

# Env. variables
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export DIFFSYNTH_MODEL_BASE_PATH="./models"
export DIFFSYNTH_SKIP_DOWNLOAD="true"

# Generation task: rap2v / rp2v / ra2v / a2v / r2v
GEN_TASK="r2v"

# Resolution mode: 720p / 480p
RESOLUTION="480p"

# Inference data
DATA_FILE="data/donghao-zhou/OmniShow_example_dataset/meta_data_infer.csv"
DATASET_BASE_PATH="data/donghao-zhou/OmniShow_example_dataset"

# Output dir
OUTPUT_DIR="outputs/infer_omnishow_wan_${GEN_TASK}_${RESOLUTION}"

# Optional: override the base diffusion model (DiT) with a fine-tuned checkpoint.
# Set the path to enable it.
DIT_CHECKPOINT=""

# Model config
MODEL_ID="Wan-AI/Wan2.1-I2V-14B-720P"
HEIGHT=1280
WIDTH=720
NUM_FRAMES=49
if [[ "$RESOLUTION" == "480p" ]]; then
  MODEL_ID="Wan-AI/Wan2.1-I2V-14B-480P"
  HEIGHT=832
  WIDTH=480
fi

# Inference config
NUM_INFERENCE_STEPS=50
CFG_SCALE=6
SEED=42

python run_infer/infer_omnishow_wan.py \
  --model_id "$MODEL_ID" \
  --csv "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --base_path "$DATASET_BASE_PATH" \
  --gen_task "$GEN_TASK" \
  --dit_checkpoint "$DIT_CHECKPOINT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --cfg_scale "$CFG_SCALE" \
  --seed "$SEED"
