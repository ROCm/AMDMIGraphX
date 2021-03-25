# BERT-SQuAD Example with MIGraphX
# Steps
1) Install MIGraphX to your environment. Please follow the steps to build MIGraphX given at https://github.com/ROCmSoftwarePlatform/AMDMIGraphX
2) Install the requirements file
```
pip3 install -r requirements_migraphx.txt
```
3) Make sure your `$LD_LIBRARY_PATH` env variable has `/opt/rocm/lib`. If not:
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib 
```  
4) Install `unzip` and fetch the uncased file (vocabulary):
```
apt-get install unzip
wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```
5) Get BERT ONNX model (bertsquad-10.onnx):
```
wget https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx
```
6) Run the inference, it will compile and run the model on three questions and a small data provided in `inputs.json`:
```
python3 bert-squad-migraphx.py
```
