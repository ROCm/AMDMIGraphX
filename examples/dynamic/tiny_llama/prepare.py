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

import argparse
import shutil
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download


def fix_batch_size(model_path: Path) -> None:
    model = onnx.load(model_path)
    values = [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ]
    for value in values:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                dim.ClearField("dim_param")
                dim.dim_value = 1
    onnx.save(model, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Download the Tiny-Llama ONNX decoder used by the dynamic example.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tiny_llama"),
        help="Directory in which to write decoder_model.onnx.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(
        repo_id="optimum-internal-testing/tiny-random-llama",
        filename="onnx_legacy/decoder_model.onnx",
        revision="merged-onnx",
    )
    destination = args.output_dir / "decoder_model.onnx"
    shutil.copyfile(source, destination)
    fix_batch_size(destination)
    print(destination)


if __name__ == "__main__":
    main()
