#!/usr/bin/env bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

set -euo pipefail

driver="${MIGRAPHX_DRIVER:-migraphx-driver}"
unset MIGRAPHX_DRIVER
model="${1:-models/tiny_llama/decoder_model.onnx}"
output="${2:-tiny_llama_dynamic.mxr}"
sequence_range='{min:1,max:64,optimals:[1,17,32,64]}'

"${driver}" compile "${model}" \
    --gpu \
    --enable-symbolic \
    --dim-param @sequence_length "${sequence_range}" \
    --output "${output}"

for sequence_length in 1 17 64; do
    "${driver}" verify "${model}" \
        --gpu \
        --compiled-model "${output}" \
        --input-dim \
        @input_ids 1 "${sequence_length}" \
        @attention_mask 1 "${sequence_length}" \
        --fill0 input_ids \
        --fill1 attention_mask
done

"${driver}" perf "${output}" \
    --gpu \
    --input-dim \
    @input_ids 1 17 \
    @attention_mask 1 17 \
    --fill0 input_ids \
    --fill1 attention_mask \
    --iterations 100
