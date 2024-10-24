#!/bin/bash

docker build --platform linux/amd64 --tag mgx_llama2:v0.1 --file Dockerfile .
