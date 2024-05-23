GPU_ID=${1:-0}

docker run -it --gpus all \
  --shm-size=10g \
  -p 5678:5678 \
  --env CUDA_VISIBLE_DEVICES=$GPU_ID \
  --env PYTHONUNBUFFERED=1 \
  photoverse:base \
  python generate.py \
  --checkpoint_path final3_photoverse_facenet_lora_rank128/photoverse_040000.pt \
  --input_image_path input_image.png \
  --guidance_scale 6 \
  --num_timesteps 25 \
  --text "a photo of a {}" \
  --negative_prompt "blurry, abstract, digital art, cartoon" \
  --num_of_samples 1