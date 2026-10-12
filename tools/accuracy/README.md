# AMD MIGraphX Accuracy checker
## Instructions
First ensure requirements and MIGraphX's python library are installed. Refer to MIGraphX instructions at the root directory to install the python library.
Use the command below to install remaining dependencies:
```
pip install -r requirements.txt
```
The accuracy checker will compare outputs from MIGraphX and onnx runtime. Therefore, an onnx file is required argument.
Example usage is below:
```
python accuracy_checker.py --onnx [path to onnx_file]
```

The output of the checker will either report as `PASSED` or `FAILED`. For detailed information,
the `--verbose` flag can be passed in to the command line which shows the mismatched elements between MIGraphX and onnx runtime.

By default, the tolerance is set to `1e-3`, but this can be changed by passing in `--tolerance [tolerance]`.
If the tolerance value is increased, then less accurate results from MIGraphX will be accepted.

For models that support variable batch sizes, use `--batch [batch_size]` to modify the batch size.

Random values are assigned to the model's inputs. However, they can be set to only contain 1s if the `--fill1` flag is passed in.
This is useful for verifying models such as bert which use integer datatypes.

By default, the CPU Execution Provider is used when running onnx runtime. If building onnx runtime with a different version, specify the provider using `--provider`.

## SSD-10 Model Zoo fixture

`ssd_10_accuracy_checker.py` compares the packaged SSD-10 test fixture with ONNX
Runtime and then compares MIGraphX with ONNX Runtime:

```bash
python ssd_10_accuracy_checker.py \
    /path/to/ssd-10 \
    --target gpu
```

## SSD-12 Model Zoo fixture

`ssd_12_accuracy_checker.py` performs the same packaged-fixture comparison for
the opset-12 SSD model:

```bash
python ssd_12_accuracy_checker.py \
    /path/to/ssd-12 \
    --target gpu
```

## Ten-image COCO SSD-10 check

First, preprocess ten JPEGs into reusable ONNX fixture directories:

```bash
python prepare_ssd_coco_inputs.py \
    /path/to/test2017 \
    /path/to/ssd_coco_test10
```

The preprocessor selects the first ten JPEG filenames in sorted order, converts
them to RGB, resizes each image to 1200x1200 with bilinear interpolation, and
applies the SSD-10 ImageNet normalization. It writes `test_data_set_0` through
`test_data_set_9` and a `manifest.json`. Pass `--overwrite` to regenerate
existing fixtures. Do not reuse the Mask R-CNN fixtures: that model uses
different resizing, color ordering, normalization, and padding.

Then compare all ten images:

```bash
python ssd_coco_accuracy_checker.py \
    /path/to/ssd-10 \
    /path/to/ssd_coco_test10 \
    --target gpu
```

The checker creates one ONNX Runtime session and compiles the fixed-shape
MIGraphX model once for all ten images. Detection counts and labels must match
exactly; boxes and scores use the requested floating-point tolerances. Use
`--ort-run` to validate only the prepared inputs and ONNX Runtime outputs.
Detections whose reference scores are equal within those tolerances are aligned
by label and box before comparison, avoiding false failures from backend rank
swaps within a numerical tie.

## Mask R-CNN Model Zoo fixture

`mask_rcnn_accuracy_checker.py` compares the packaged Mask R-CNN test fixture
with ONNX Runtime and then compares MIGraphX with ONNX Runtime:

```bash
python mask_rcnn_accuracy_checker.py \
    /path/to/mask_rcnn_R_50_FPN_1x \
    --target gpu
```

## Ten-image COCO Mask R-CNN check

The COCO check uses two scripts. First, preprocess ten JPEGs into reusable ONNX
fixture directories:

```bash
python prepare_mask_rcnn_coco_inputs.py \
    /path/to/test2017 \
    /path/to/mask_rcnn_coco_test10
```

The preprocessor selects the first ten JPEG filenames in sorted order. It
resizes the shorter side to 800 pixels, converts RGB to BGR, subtracts
`[102.9801, 115.9465, 122.7717]`, and pads the spatial dimensions to multiples
of 32. It writes `test_data_set_0` through `test_data_set_9` and a
`manifest.json`. Pass `--overwrite` to regenerate existing fixtures.

Then compare all ten images:

```bash
python mask_rcnn_coco_accuracy_checker.py \
    /path/to/mask_rcnn_R_50_FPN_1x \
    /path/to/mask_rcnn_coco_test10 \
    --target gpu
```

The checker creates one ONNX Runtime session and groups cases by prepared input
shape. It compiles and runs one MIGraphX shape group at a time, then releases
that model before compiling the next shape. All ten detection counts and boxes,
labels, scores, and masks must match for the command to pass. Use `--ort-run`
to validate only the prepared inputs and ONNX Runtime outputs.
