#!/bin/bash

model_dir=$ONNX_HOME/.onnx/models
tmp_dir=$ONNX_HOME/tmp/
mkdir -p $model_dir
mkdir -p $tmp_dir
for name in bvlc_alexnet densenet121 inception_v1 inception_v2 \
        resnet50 shufflenet squeezenet vgg19 zfnet512 
do
curl https://s3.amazonaws.com/download.onnx/models/opset_9/$name.tar.gz --output $tmp_dir/$name.tar.gz
tar -xzvf $tmp_dir/$name.tar.gz --directory $model_dir && rm $tmp_dir/$name.tar.gz
done

