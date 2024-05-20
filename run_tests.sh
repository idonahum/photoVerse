#!/bin/bash

# Define paths
MODEL1_PATH="photoverse_058000.pt"
MODEL2_PATH="photoverse_arcface_042000.pt"
MODEL3_PATH="photoverse_facenet_074000.pt"

# Define other common arguments
PRETRAINED_MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_ROOT_PATH="train_test_from_file"
IMG_SUBFOLDER="images"
OUTPUT_DIR="results"
BATCH_SIZE=16
NUM_WORKERS=4
GUIDANCE_SCALE=2.0
DEVICE="cuda"  # or "cpu"
RESOLUTION=512
ARCFACE_MODEL_ROOT_DIR="arcface_model"

# Define timesteps
TIMESTEPS=(10 25 50 100)

# Run the script for each model and each timestep
for TIMESTEP in "${TIMESTEPS[@]}"; do
    for MODEL_PATH in "$MODEL1_PATH" "$MODEL2_PATH" "$MODEL3_PATH"; do
        python test.py --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME \
                              --pretrained_photoverse_path $MODEL_PATH \
                              --data_root_path $DATA_ROOT_PATH \
                              --img_subfolder $IMG_SUBFOLDER \
                              --output_dir $OUTPUT_DIR \
                              --batch_size $BATCH_SIZE \
                              --num_workers $NUM_WORKERS \
                              --denoise_timesteps $TIMESTEP \
                              --guidance_scale $GUIDANCE_SCALE \
                              --device $DEVICE \
                              --resolution $RESOLUTION \
                              --arcface_model_root_dir $ARCFACE_MODEL_ROOT_DIR \
                              --max_gen_images 16
    done
done
