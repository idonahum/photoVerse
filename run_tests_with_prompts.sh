#!/bin/bash

# Define paths
MODEL_PATH="photoverse_facenet_074000.pt"

# Define other common arguments
PRETRAINED_MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_ROOT_PATH="train_test_from_file"
IMG_SUBFOLDER="images"
OUTPUT_DIR="results"
BATCH_SIZE=32
NUM_WORKERS=4
GUIDANCE_SCALE=2.0
DEVICE="cuda"  # or "cpu"
RESOLUTION=512
MAX_GEN_IMAGES=96
DENOISE_TIMESTEPS=25  # Update if you want to run for multiple timesteps

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Run the script for each model
python test_prompts.py --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME \
                      --pretrained_photoverse_path $MODEL_PATH \
                      --data_root_path $DATA_ROOT_PATH \
                      --img_subfolder $IMG_SUBFOLDER \
                      --output_dir $OUTPUT_DIR \
                      --batch_size $BATCH_SIZE \
                      --num_workers $NUM_WORKERS \
                      --denoise_timesteps $DENOISE_TIMESTEPS \
                      --guidance_scale $GUIDANCE_SCALE \
                      --device $DEVICE \
                      --resolution $RESOLUTION \
                      --max_gen_images $MAX_GEN_IMAGES
