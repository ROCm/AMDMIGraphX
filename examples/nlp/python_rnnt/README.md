# RNN-T

This version was taken from [MLCommons](https://github.com/mlcommons/inference/tree/master/retired_benchmarks/speech_recognition/rnnt)

## Run Script

To run the RNN-T, follow these steps below.

Download model inference and checkpoints via [run.sh](./run.sh) 
**Note: Run this outside of docker. Will take longer to download within docker.***

```bash
bash run.sh
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the script.

```bash
python3 rnnt.py 
```

This will do three things:
- Run the PyTorch model 
- Export the PyTorch model to ONNX
- Use MIGraphX to read the ONNX model 
- Run the MIGraphX model 

Transcript: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel 

PyTorch Output: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel

MIGX Output: miso quter is the aposile o the midle clases an we arad to welcome his gospel