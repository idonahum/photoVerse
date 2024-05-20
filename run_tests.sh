#!/bin/bash

# Define paths
MODEL1_PATH="path/to/model1"
MODEL2_PATH="path/to/model2"
MODEL3_PATH="path/to/model3"

# Define other common arguments
PRETRAINED_MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_ROOT_PATH="path/to/data_root"
IMG_SUBFOLDER="images"
OUTPUT_DIR="results"
BATCH_SIZE=16
NUM_WORKERS=4
GUIDANCE_SCALE=2.0
DEVICE="cuda"  # or "cpu"
EXTRA_NUM_TOKENS=4
IMAGE_ENCODER_LAYERS_IDX="[4, 8, 12, 16]"
RESOLUTION=512
ARCFACE_MODEL_ROOT_DIR="arcface_model"

# Define timesteps
TIMESTEPS=(10 25 50 100)

# Run the script for each model and each timestep
for TIMESTEP in "${TIMESTEPS[@]}"; do
    for MODEL_PATH in "$MODEL1_PATH" "$MODEL2_PATH" "$MODEL3_PATH"; do
        python main_script.py --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME \
                              --pretrained_photoverse_path $MODEL_PATH \
                              --data_root_path $DATA_ROOT_PATH \
                              --img_subfolder $IMG_SUBFOLDER \
                              --output_dir $OUTPUT_DIR \
                              --batch_size $BATCH_SIZE \
                              --num_workers $NUM_WORKERS \
                              --denoise_timesteps $TIMESTEP \
                              --guidance_scale $GUIDANCE_SCALE \
                              --device $DEVICE \
                              --extra_num_tokens $EXTRA_NUM_TOKENS \
                              --image_encoder_layers_idx $IMAGE_ENCODER_LAYERS_IDX \
                              --resolution $RESOLUTION \
                              --arcface_model_root_dir $ARCFACE_MODEL_ROOT_DIR
    done
done
