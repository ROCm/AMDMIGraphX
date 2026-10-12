#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""Prepare ten COCO JPEGs for the ONNX Model Zoo SSD-10 model."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from onnx import numpy_helper
from PIL import Image

INPUT_NAME = "image"
IMAGE_COUNT = 10
INPUT_SHAPE = (1, 3, 1200, 1200)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
STDDEV = np.asarray([0.229, 0.224, 0.225],
                    dtype=np.float32)[None, :, None, None]


def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Preprocess ten COCO JPEGs into ONNX TensorProto fixtures "
        "for SSD-10."))
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Extracted COCO test2017 directory containing JPEG images",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=
        "Directory in which test_data_set_0 through test_data_set_9 are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated cases and manifest",
    )
    return parser.parse_args()


def select_images(images_dir):
    images = sorted(images_dir.glob("*.jpg"))
    if len(images) < IMAGE_COUNT:
        raise ValueError(
            f"Expected at least {IMAGE_COUNT} JPEGs in {images_dir}, found {len(images)}"
        )
    return images[:IMAGE_COUNT]


def preprocess_image(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        image = image.resize((1200, 1200), Image.Resampling.BILINEAR)

    image = np.asarray(image, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = (image / np.float32(255.0) - MEAN) / STDDEV
    result = np.ascontiguousarray(image, dtype=np.float32)
    return result, {
        "original_size": [original_height, original_width],
        "resized_size": [1200, 1200],
        "tensor_shape": list(result.shape),
    }


def prepare_output(output_dir, overwrite):
    manifest_path = output_dir / "manifest.json"
    case_dirs = [
        output_dir / f"test_data_set_{index}" for index in range(IMAGE_COUNT)
    ]
    existing = [path for path in [manifest_path, *case_dirs] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Generated output already exists at {existing[0]}; use --overwrite"
        )
    if overwrite:
        for case_dir in case_dirs:
            if case_dir.is_dir():
                shutil.rmtree(case_dir)
            elif case_dir.exists():
                case_dir.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return manifest_path, case_dirs


def write_tensor(path, data):
    tensor = numpy_helper.from_array(data, name=INPUT_NAME)
    path.write_bytes(tensor.SerializeToString())


def main():
    args = parse_args()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(
            f"Missing COCO image directory: {args.images_dir}")

    images = select_images(args.images_dir)
    manifest_path, case_dirs = prepare_output(args.output_dir, args.overwrite)
    cases = []
    for index, (image_path, case_dir) in enumerate(zip(images, case_dirs)):
        tensor, metadata = preprocess_image(image_path)
        if tensor.shape != INPUT_SHAPE:
            raise ValueError(
                f"Unexpected preprocessed shape {tensor.shape} for {image_path}"
            )
        case_dir.mkdir()
        write_tensor(case_dir / "input_0.pb", tensor)
        case = {
            "case": case_dir.name,
            "image_id": image_path.stem,
            "source_file": image_path.name,
            **metadata,
        }
        cases.append(case)
        print(f"[{index + 1}/{IMAGE_COUNT}] {image_path.name} -> "
              f"{case_dir.name}/input_0.pb {tuple(tensor.shape)}")

    manifest = {
        "input_name": INPUT_NAME,
        "image_count": IMAGE_COUNT,
        "preprocessing": {
            "resize": [1200, 1200],
            "color_order": "RGB",
            "scale": "0..1",
            "mean": MEAN.reshape(-1).tolist(),
            "stddev": STDDEV.reshape(-1).tolist(),
        },
        "cases": cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {IMAGE_COUNT} cases in {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
