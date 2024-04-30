GPU_ID=${1:-0}

docker run -d --gpus all \
  --shm-size=10g \
  -v /home/lab/haimzis/projects/photoVerse:/workspace \
  -v /home:/home \
  -v /cortex/users/haimzis:/cortex/users/haimzis \
  -v /cortex/users/haimzis/.cache:/root/.cache/ \
  --env CUDA_VISIBLE_DEVICES=$GPU_ID \
  --env PYTHONUNBUFFERED=1 \
  photoverse:base \
  ./prepare_dataset_and_train.sh
