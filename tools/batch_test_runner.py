#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
import sys
import os
import subprocess
import argparse
from dataclasses import dataclass

sys.path.append(os.environ["PYTHONPATH"])


@dataclass
class Command:
    filename: str
    options: str


COMMAND_LIST = [
    Command("opset7/fp16_inception_v1", "--atol 4e-2 --rtol 4e-2"),
    Command("opset7/test_bvlc_googlenet", None),
    Command("opset7/test_emotion_ferplus", None),
    Command("opset7/test_mobilenetv2-1.0", None),
    Command("opset7/test_resnet34v2", None),
    Command("opset7/test_squeezenet", None),
    Command("opset7/test_zfnet512", None),
    Command("opset7/tf_inception_v3", None),
    Command("opset7/tf_mobilenet_v2_1.4_224", None),
    Command("opset7/tf_resnet_v1_101", None),
    Command("opset7/tf_resnet_v2_152", None),
    Command("opset7/fp16_shufflenet", None),
    Command("opset7/test_bvlc_reference_caffenet", None),
    Command("opset7/test_inception_v1", None),
    Command("opset7/test_resnet101v2", None),
    Command("opset7/test_resnet50", None),
    Command("opset7/test_squeezenet1.1", None),
    Command("opset7/tf_inception_resnet_v2", None),
    Command("opset7/tf_inception_v4", None),
    Command("opset7/tf_nasnet_large", None),
    Command("opset7/tf_resnet_v1_152", None),
    Command("opset7/tf_resnet_v2_50", None),
    Command("opset7/fp16_tiny_yolov2", "--atol 4e-2 --rtol 4e-2"),
    Command("opset7/test_bvlc_reference_rcnn_ilsvrc13", None),
    Command("opset7/test_inception_v2", None),
    Command("opset7/test_resnet152v2", None),
    Command("opset7/test_resnet50v2", None),
    Command("opset7/test_tiny_yolov2", None),
    Command("opset7/tf_inception_v1", None),
    Command("opset7/tf_mobilenet_v1_1.0_224", None),
    Command("opset7/tf_nasnet_mobile", None),
    Command("opset7/tf_resnet_v1_50", None),
    Command("opset7/test_bvlc_alexnet", None),
    Command("opset7/test_densenet121", None),
    Command("opset7/test_mnist", None),
    Command("opset7/test_resnet18v2", None),
    Command("opset7/test_shufflenet", None),
    Command("opset7/test_vgg19", None),
    Command("opset7/tf_inception_v2", None),
    Command("opset7/tf_mobilenet_v2_1.0_224", None),
    Command("opset7/tf_pnasnet_large", None),
    Command("opset7/tf_resnet_v2_101", None),
    Command("opset8/fp16_inception_v1", "--atol 4e-2 --rtol 4e-2"),
    Command("opset8/fp16_shufflenet", "--atol 4e-2 --rtol 4e-2"),
    Command("opset8/fp16_tiny_yolov2", "--atol 4e-2 --rtol 4e-2"),
    Command("opset8/mxnet_arcface", None),
    Command("opset8/test_bvlc_alexnet", None),
    Command("opset8/test_bvlc_googlenet", None),
    Command("opset8/test_bvlc_reference_caffenet", None),
    Command("opset8/test_bvlc_reference_rcnn_ilsvrc13", None),
    Command("opset8/test_densenet121", None),
    Command("opset8/test_emotion_ferplus", None),
    Command("opset8/test_inception_v1", None),
    Command("opset8/test_inception_v2", None),
    Command("opset8/test_mnist", None),
    Command("opset8/test_resnet50", None),
    Command("opset8/test_shufflenet", None),
    Command("opset8/test_squeezenet", None),
    Command("opset8/test_tiny_yolov2", None),
    Command("opset8/test_vgg19", None),
    Command("opset8/test_zfnet512", None),
    Command("opset8/tf_inception_resnet_v2", None),
    Command("opset8/tf_inception_v1", None),
    Command("opset8/tf_inception_v2", None),
    Command("opset8/tf_inception_v3", None),
    Command("opset8/tf_inception_v4", None),
    Command("opset8/tf_mobilenet_v1_1.0_224", None),
    Command("opset8/tf_mobilenet_v2_1.0_224", None),
    Command("opset8/tf_mobilenet_v2_1.4_224", None),
    Command("opset8/tf_nasnet_large", None),
    Command("opset8/tf_nasnet_mobile", None),
    Command("opset8/tf_pnasnet_large", None),
    Command("opset8/tf_resnet_v1_101", None),
    Command("opset8/tf_resnet_v1_152", None),
    Command("opset8/tf_resnet_v1_50", None),
    Command("opset8/tf_resnet_v2_101", None),
    Command("opset8/tf_resnet_v2_152", None),
    Command("opset8/tf_resnet_v2_50", None),
    Command("opset9/LSTM_Seq_lens_unpacked", None),
    Command("opset9/candy", None),
    Command("opset9/cgan", None),
    Command("opset9/mosaic", None),
    Command("opset9/pointilism", None),
    Command("opset9/rain_princess", None),
    Command("opset9/tf_inception_resnet_v2", None),
    Command("opset9/tf_inception_v1", None),
    Command("opset9/tf_inception_v2", None),
    Command("opset9/tf_inception_v3", None),
    Command("opset9/tf_inception_v4", None),
    Command("opset9/tf_mobilenet_v1_1.0_224", None),
    Command("opset9/tf_mobilenet_v2_1.0_224", None),
    Command("opset9/tf_mobilenet_v2_1.4_224", None),
    Command("opset9/tf_nasnet_large", None),
    Command("opset9/tf_nasnet_mobile", None),
    Command("opset9/tf_pnasnet_large", None),
    Command("opset9/tf_resnet_v1_101", None),
    Command("opset9/tf_resnet_v1_152", None),
    Command("opset9/tf_resnet_v1_50", None),
    Command("opset9/tf_resnet_v2_101", None),
    Command("opset9/tf_resnet_v2_152", None),
    Command("opset9/tf_resnet_v2_50", None),
    Command("opset9/udnie", None),
    Command("opset10/tf_nasnet_large", None),
    Command("opset11/tf_inception_resnet_v2", None),
    Command("opset11/tf_inception_v1", None),
    Command("opset11/tf_inception_v2", None),
    Command("opset11/tf_inception_v3", None),
    Command("opset11/tf_inception_v4", None),
    Command("opset11/tf_mobilenet_v1_1.0_224", None),
    Command("opset11/tf_mobilenet_v2_1.0_224", None),
    Command("opset11/tf_mobilenet_v2_1.4_224", None),
    Command("opset11/tf_nasnet_large", None),
    Command("opset11/tf_nasnet_mobile", None),
    Command("opset11/tf_pnasnet_large", None),
    Command("opset11/tf_resnet_v1_101", None),
    Command("opset11/tf_resnet_v1_152", None),
    Command("opset11/tf_resnet_v1_50", None),
    Command("opset11/tf_resnet_v2_101", None),
    Command("opset11/tf_resnet_v2_152", None),
    Command("opset11/tf_resnet_v2_50", None),
    Command("opset11/tinyyolov3", None),
]
TEST_RUNNER = "test_runner.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runs MIGX test runner on all tests in a directory")
    parser.add_argument('test_dir',
                        type=str,
                        help='folder where the tests are stored')
    args = parser.parse_args()
    for c in COMMAND_LIST:
        command_list = [
            "python3",
            f"{os.path.join(os.getcwd(), TEST_RUNNER)}",
            f"{os.path.join(args.test_dir, c.filename)}",
        ]
        if c.options:
            options = c.options.split()
            for o in options:
                command_list.append(o)
        subprocess.run(command_list, check=True)


if __name__ == "__main__":
    sys.exit(main())
