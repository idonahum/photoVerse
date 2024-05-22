GPU_ID=${1:-0}

docker run -it --gpus all \
  --shm-size=10g \
  -v /home/lab/haimzis/projects/photoVerse:/workspace \
  -v /home:/home \
  -v /cortex/users/haimzis:/cortex/users/haimzis \
  -v /cortex/users/haimzis/.cache:/root/.cache/ \
  -p 5678:5678 \
  --env CUDA_VISIBLE_DEVICES=$GPU_ID \
  --env PYTHONUNBUFFERED=1 \
  photoverse:base \
  python generate.py \
  --checkpoint_path /home/lab/haimzis/projects/photoVerse/runs/final3_photoverse_facenet_lora_rank128/photoverse_040000.pt \
  --input_image_path 1.jpg \
  --guidance_scale 2 \
  --num_timesteps 20 \
  --text "image of {}"