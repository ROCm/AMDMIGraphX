# ResNet-50 python parsing example

This sample uses the MGX common api python module to parse, optimize and perform inference with a ResNet-50 model saved in the ONNX format.

## Prerequisites

This sample requires the hip-python package to be installed
```bash
# x.y.z should be replaced by the first three digits of the ROCm installation version number
python3 -m pip install -i https://test.pypi.org/simple hip-python~=x.y.z
```
To install other required packages run
```bash
pip install -r requirements.txt
```
For the sample to be, the branch it is part of must first be built, after which the migraphx python module must be made visible to the python interpreter. 
This can be done in one of two ways:
```bash
# The two paths assume the migraphx docker is being used, if not they must be appropriately adjusted
export PYTHONPATH=$PYTHONPATH:/code/AMDMIGraphX/build/lib
export LD_LIB_PATH=$LD_LIB_PATH:/code/AMDMIGraphX/build/lib
```
Alternately
```bash
#from within the build directory
make install

export PYTHONPATH=$PYTHONPATH:/opt/rocm/lib
export LD_LIB_PATH=$LD_LIB_PATH:/opt/rocm/lib
```

## Running the sample
To run the sample, switch the current working directory to src/common_api/samples/python/introductory_parser_samples and invoke it with the python interpreter
```bash
python3 onnx_resnet50.py
```
By default, the sample will search for `data` in /code/AMDMIGraphX/src/common_api/samples_data. A custom path can be provided by using the `-d` option, e.g.:  `python3 onnx_resnet50.py -d /custom/path/to/data`

