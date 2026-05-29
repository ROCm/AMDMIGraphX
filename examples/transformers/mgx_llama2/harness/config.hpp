/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

const std::string MODEL_FOLDER   = "/model/";
const std::string ONNX_FILE      = "model.onnx";
const std::string DATASET_FOLDER = "/dataset/";
const size_t DATASET_SIZE        = 10;
const size_t BATCH_SIZE          = 1;
// sequence length from model config
const size_t SEQ_SIZE = 1024;
// vocab size from model config
const size_t VOCAB_SIZE = 32000;
// EOS token from model config
const size_t EOS = 2;
// Write output tokens to file
const bool WRITE_RESULT_FILE = false;

const int DEVICE_ID = 4;

const size_t HIDDEN_LAYERS_NUM = 32;
const size_t HEAD_SIZE         = 128;
const size_t PAST_KEY_VAL_SIZE = BATCH_SIZE * HIDDEN_LAYERS_NUM * HEAD_SIZE * SEQ_SIZE;