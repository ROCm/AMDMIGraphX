# Object Detection and Segmentation Accuracy Examples

These examples compare MIGraphX inference with ONNX Runtime and the packaged
reference outputs for two ONNX Model Zoo models:

- Mask R-CNN-10 instance segmentation
- SSD-12 object detection

The scripts download the model and its `test_data_set_0` fixture from the
[ONNX Model Zoo mirror](https://github.com/mitzen/onnx-zoo-models/tree/main/vision/object_detection_segmentation).
The downloads are verified against their Git LFS SHA-256 identifiers and
extracted under `~/.cache/migraphx/model_zoo` by default. No model or test data
is stored in the MIGraphX source tree.

## Requirements

Install MIGraphX with its Python module, then install the remaining packages:

```bash
python3 -m pip install -r requirements.txt
```

## Mask R-CNN

The first run downloads about 165 MB:

```bash
python3 mask_rcnn_accuracy_checker.py
```

## SSD-12

The first run downloads about 91 MB:

```bash
python3 ssd_12_accuracy_checker.py
```

Both scripts compile for the GPU target by default. Use `--target ref` to run
with the reference target, `--ort-run` to validate only ONNX Runtime and the
packaged outputs, or `--cache-dir PATH` to select another download location.
Run either script with `--help` for all options.
