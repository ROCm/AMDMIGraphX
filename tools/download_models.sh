#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

if [ -z "$ONNX_HOME" ]
then
   ONNX_HOME=$HOME
fi

model_dir=$ONNX_HOME/.onnx/models
tmp_dir=$ONNX_HOME/tmp/
mkdir -p $model_dir
mkdir -p $tmp_dir
models="bvlc_alexnet \
        densenet121 \
        inception_v2 \
        shufflenet \
        vgg19 \
        zfnet512"

#for name in $models
#do
# TODO : temporarily disable model downloads to get CI running again
# See https://github.com/onnx/onnx/issues/4857 for problem
#curl https://s3.amazonaws.com/download.onnx/models/opset_9/$name.tar.gz --output $tmp_dir/$name.tar.gz
#tar -xzvf $tmp_dir/$name.tar.gz --directory $model_dir && rm $tmp_dir/$name.tar.gz
#done
