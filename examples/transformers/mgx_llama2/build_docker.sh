#!/bin/bash

GPU_TARGET=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')

docker build --platform linux/amd64 --tag mgx_llama2:v0.2 --build-arg GPU_TARGET=$GPU_TARGET --file Dockerfile .
