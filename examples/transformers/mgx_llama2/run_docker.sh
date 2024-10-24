#!/bin/bash

if [[ -z "${MODEL_DIR_PATH}" ]]; then
  echo "MODEL_DIR_PATH is not set, please provide the path to model before running docker."
  exit 1
else
  MODEL_DIR="${MODEL_DIR_PATH}"
fi

docker run --device='/dev/kfd' --device='/dev/dri' --group-add video \
-v $MODEL_DIR:/model \
-w /mgx_llama2/build \
-it mgx_llama2:v0.1

