# Models

## UFF

resnet50-infer-5.uff
- trained by NVidia, based on ResNet50 V1 model from [TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim)
- converted to UFF using `convert-to-uff`
  - `convert-to-uff <models>/resnet_all-nlayer_50__precision0_randominit.pb -o tf2trt_resnet50.uff -t -O spatial_avg`

## Caffe

ResNet50_N2.prototxt and ResNet50_fp32.caffemodel
- downloaded from https://github.com/KaimingHe/deep-residual-networks#models
