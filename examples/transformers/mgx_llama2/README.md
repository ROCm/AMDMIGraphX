## Getting the model

### Getting the pre-quantized model from HuggingFace
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login YOUR_HF_TOKEN
hugginggface-cli download https://huggingface.co/amd/Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx
```
Alternatively you can quantize the model yourself.

### Quantizing the model

**If you are using the pre-quantized model you can skip this section.**

Get the latest quark quantizer version from https://xcoartifactory/ui/native/uai-pip-local/com/amd/quark/main/nightly/ . Downloading the zip is recommended because it contains the required scripts. The quark version used when this was created: quark-1.0.0.dev20241028+eb46b7438 (28-10-24).

Also we will need to install the onnxruntime-genai (OGA) tool to convert the quark_safetensors format to onnx format properly. 

#### Installing quark and it's dependencies:
```bash
# install OGA tool
pip install onnxruntime-genai

# Quark dependencies according to https://quark.docs.amd.com/latest/install.html, we assume pytorch is already installed. You can use the following base docker image which has torch installed: rocm/pytorch:rocm6.2.2_ubuntu20.04_py3.9_pytorch_release_2.2.1 
pip install onnxruntime onnx

# Install the whl
unzip quark-1.0.0.dev20241028+eb46b7438.zip -d quark
cd quark
RUN pip install quark-1.0.0.dev20241028+eb46b7438-py3-none-any.whl
```

#### Quantizing the model and converting to ONNX
```bash
cd quark/examples/torch/language_modeling/llm_ptq

export MODEL_DIR = [local model checkpoint folder] or meta-llama/Llama-2-7b-chat-hf or meta-llama/Llama-2-70b-chat-hf
export QUANTIZED_MODEL_DIR = [output model checkpoint folder]

python3 quantize_quark.py --model_dir $MODEL_DIR \
                          --data_type float16 \
                          --quant_scheme w_uint4_per_group_asym \
                          --num_calib_data 128 \
                          --quant_algo awq \
                          --dataset pileval_for_awq_benchmark \
                          --seq_len 1024 \
                          --output_dir $MODEL_DIR-awq-uint4-asym-g128-f16 \
                          --model_export quark_safetensors \
                          --custom_mode awq

python3 -m onnxruntime_genai.models.builder \
            -i "$QUANTIZED_MODEL_DIR" \
            -o "$QUANTIZED_MODEL_DIR-onnx" \
            -p int4 \
            -e cpu
```

## Getting the dataset

Download the preprocessed open-orca dataset files using the instructions in https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed

### Running the example

#### Starting migraphx docker

```bash

./build_docker.sh

export MODEL_DIR_PATH=path/to/quantized/llama2-7[0]b-model
export DATA_DIR_PATH=path/to/open_orca_dataset
./run_docker.sh
```

#### Building and running the example

```bash
# Convert dataset to numpy format
./prepocess_dataset.py

# Builidng the example
cd mgx_llama2
mkdir build && cd build
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
make -j

# Running the example
export MIOPEN_FIND_ENFORCE=3
./mgxllama2

# Test the accuracy of the output
python3 eval_accuracy.py
```
