# Object Detection and Segmentation Accuracy Examples

These examples compare MIGraphX inference with ONNX Runtime for three ONNX
Model Zoo models:

- Mask R-CNN-10 instance segmentation
- SSD-10 object detection
- SSD-12 object detection

The packaged-fixture examples download the model and its `test_data_set_0` from the
[ONNX Model Zoo mirror](https://github.com/mitzen/onnx-zoo-models/tree/main/vision/object_detection_segmentation).
The COCO examples additionally download the first ten sorted JPEGs from COCO
`test2017` and apply each model's preprocessing in memory. All downloads are
verified against fixed SHA-256 values and cached under
`~/.cache/migraphx/model_zoo` by default. No model, test data, or COCO image is
stored in the MIGraphX source tree.

## Requirements

Install MIGraphX with its Python module, then install the remaining packages:

```bash
python3 -m pip install -r requirements.txt
```

## Packaged Model Zoo fixtures

The first Mask R-CNN run downloads about 165 MB:

```bash
python3 mask_rcnn_accuracy_checker.py
```

The first SSD-12 run downloads about 91 MB:

```bash
python3 ssd_12_accuracy_checker.py
```

Each script compares ONNX Runtime with the packaged outputs before comparing
MIGraphX with ONNX Runtime.

## Ten-image COCO checks

The two COCO checks share the same ten `test2017` JPEGs. The image download is
1,873,441 bytes (about 1.87 MB) and happens automatically on the first run.

Mask R-CNN resizes each image so its shorter side is 800 pixels, converts RGB to
BGR, subtracts the model mean, and pads the spatial dimensions to multiples of
32. Because the resulting dimensions vary, MIGraphX compiles once per distinct
input shape:

```bash
python3 mask_rcnn_coco_accuracy_checker.py
```

SSD-10 resizes every image to 1200x1200 and applies ImageNet normalization, so
MIGraphX compiles the fixed-shape model once:

```bash
python3 ssd_coco_accuracy_checker.py
```

The images remain subject to their individual
[COCO source licenses](https://cocodataset.org/#termsofuse), which include
attribution, noncommercial, and no-derivatives variants. The scripts retrieve
the images from the official COCO server; MIGraphX does not redistribute them.

All scripts compile for the GPU target by default. Use `--target ref` to run
with the reference target, `--ort-run` to run only ONNX Runtime, or
`--cache-dir PATH` to select another download location. Run a script with
`--help` for all options.
