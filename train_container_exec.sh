GPU_ID=${1:-0}

docker run -d --gpus all \
  --shm-size=10g \
  --env CUDA_VISIBLE_DEVICES=$GPU_ID \
  --env PYTHONUNBUFFERED=1 \
  photoverse:base \
  ./prepare_dataset_and_train.sh
