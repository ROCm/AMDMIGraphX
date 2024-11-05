#!/bin/bash

docker build --platform linux/amd64 --tag mgx_llama2:v0.2 --file Dockerfile .
