# Setting Up MNIST Samples

## Models

mnist.onnx: Opset 8, Retrieved from [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/classification/mnist)

## Run ONNX model with trtexec

* FP32 precisons with fixed batch size 1
  * `./trtexec --explicitBatch --onnx=mnist.onnx --workspace=1024`
* Other precisions
  * Add `--fp16` for FP16 and `--int8` for INT8.

## Run safety ONNX model with sampleSafeMNIST

* Build safe engine
  * `./sample_mnist_safe_build`
* Inference
  * `./sample_mnist_safe_infer`
* See sample READEME for more details.
