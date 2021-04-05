# Super Resolution with AMD MIGraphX

This example is based on [ONNX run_super_resolution_model notebook](https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb) and modified for MIGraphX.

## Description
Given an input image, this application resizes the image to 224x224 and then scales it to 672x672, thus it is useful for upscaling low-resolution images.

### Model Utilized
> "Super Resolution uses efficient  [Sub-pixel convolutional layer](https://arxiv.org/abs/1609.05158) described for increasing spatial resolution within network tasks. By increasing pixel count, images are then clarified, sharpened, and upscaled without losing the input image’s content and characteristics." [[Reference]](https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/README.md)

Model in PyTorch definitions:
```
self.relu = nn.ReLU(inplace=inplace)
self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
```
## How-to
Run Jupyter notebook server on a ROCm and MIGraphX installed system, and run `Run_Super_Resolution_Model.ipynb`