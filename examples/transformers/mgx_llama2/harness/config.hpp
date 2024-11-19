#pragma once

// TODO: fix paths
const std::string MODEL_FOLDER = "/model/";
const std::string ONNX_FILE = "model.onnx";
const std::string DATASET_FOLDER = "/dataset/";
const size_t DATASET_SIZE = 10;
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
const size_t HEAD_SIZE = 128;
const size_t PAST_KEY_VAL_SIZE = HIDDEN_LAYERS_NUM*HEAD_SIZE*SEQ_SIZE;