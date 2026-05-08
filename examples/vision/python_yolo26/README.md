# YOLO26 Object Detection
The notebook [yolo26_inference.ipynb](./yolo26_inference.ipynb) is intended to be an example of how to use MIGraphX to perform object detection. The model used within is a pre-trained YOLO26 from Ultralytics, which is exported to ONNX format and compiled with AMD MIGraphX (applying FP16 quantization).

## Run the Notebook
To run the example notebook, simply issue the following command from this directory:
```
$ jupyter notebook yolo26_inference.ipynb
```
