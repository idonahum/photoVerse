output=$(python prepare_celebhqmasks.py)

# Parse the output to extract train_folder, test_folder, and save_path
train_folder=$(echo "$output" | awk '/Train folder:/ {print $NF}')
test_folder=$(echo "$output" | awk '/Test folder:/ {print $NF}')
#arcface_models_path=$(echo "$output" | awk '/ArcFace models folder:/ {print $NF}')

CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file single_gpu.json train.py --data_root_path $train_folder --mask_subfolder masks --output_dir exp1 --max_train_steps 40000 --train_batch_size 4 --report_to wandb --save_steps 2000
