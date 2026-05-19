#!/usr/bin/env bash

set -euo pipefail

# Env. variables
export PYTHONNOUSERSITE=1
# export PORT=48953  # Uncomment to explicitly avoid port conflict
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
export DIFFSYNTH_MODEL_BASE_PATH="./models"
export DIFFSYNTH_SKIP_DOWNLOAD="true"

# Generation task: rap2v / rp2v / ra2v / a2v / r2v
GEN_TASK="r2v"

# Resolution mode: 720p / 480p
RESOLUTION="480p"

# Train data
DATA_FILE="data/donghao-zhou/OmniShow_example_dataset/meta_data_train.csv"
DATASET_BASE_PATH="data/donghao-zhou/OmniShow_example_dataset"

# Output dir
OUTPUT_DIR="ckpts/train_omnishow_wan_${GEN_TASK}_${RESOLUTION}"

# Train config
LAUNCH_NUM_PROCESSES=8  # match your GPU count for training
ACCELERATE_CONFIG="run_train/accelerate_config_14B_zero3.yaml"
MODEL_ID="Wan-AI/Wan2.1-I2V-14B-720P"
if [[ "$RESOLUTION" == "480p" ]]; then
  MODEL_ID="Wan-AI/Wan2.1-I2V-14B-480P"
fi
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ID}:Wan2.1_VAE.pth,${MODEL_ID}:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# Key parameters
LEARNING_RATE="1e-5"  # for normal params
LEARNING_RATE_AUDIO="1e-4"  # for audio-related params
NUM_EPOCHS=5000
SAVE_STEPS=1000
NUM_FRAMES=49
HEIGHT=1280
WIDTH=720
if [[ "$RESOLUTION" == "480p" ]]; then
  HEIGHT=832
  WIDTH=480
fi


accelerate launch --config_file "$ACCELERATE_CONFIG" \
  --debug \
  --num_processes "$LAUNCH_NUM_PROCESSES" \
  --main_process_port "$PORT" \
  run_train/train_omnishow_wan.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATA_FILE" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --gen_task "$GEN_TASK" \
  --model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN_PATHS" \
  --trainable_models "dit" \
  --learning_rate "$LEARNING_RATE" \
  --learning_rate_audio "$LEARNING_RATE_AUDIO" \
  --num_epochs "$NUM_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu
