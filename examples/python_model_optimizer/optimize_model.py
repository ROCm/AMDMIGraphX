import migraphx
import argparse
import os

parser = argparse.ArgumentParser(
    description="Compile and serialize model offline for future deployment.")
parser.add_argument("model", type=str, help="Path to ONNX or TF Protobuf file")
parser.add_argument("--nchw",
                    help="Treat Tensorflow format as nchw. (Default is nhwc)",
                    action="store_true")
parser.add_argument("-t",
                    "--target",
                    type=str,
                    choices=["gpu", "cpu", "ref"],
                    help="Compilation target")
parser.add_argument("-d",
                    "--disable_fast_math",
                    help="Disable optimized math functions",
                    action="store_true")
parser.add_argument("-o",
                    "--offload_copy_off",
                    help="Disable implicit offload copying",
                    action="store_true")
parser.add_argument("-f",
                    "--fp16",
                    help="Quantize model in FP16 precision",
                    action="store_true")
parser.add_argument("-i",
                    "--int8",
                    help="Quantize model in INT8 precision",
                    action="store_true")
parser.add_argument(
    "-j",
    "--json",
    help="Save program in JSON format. (Default is MsgPack format)",
    action="store_true")
parser.add_argument(
    "-p",
    "--output_path",
    type=str,
    help=
    "Specify file name and/or path for model to be saved. (Default is ./saved_models/<model_name>.<format>"
)
args = parser.parse_args()

model_name = args.model
if not os.path.isabs(model_name):
    dirname = os.path.abspath(os.path.dirname(__file__))
    model_name = os.path.join(dirname, model_name)
    model_name = os.path.abspath(model_name)

if ".onnx" in model_name:
    model = migraphx.parse_onnx(model_name)
elif ".pb" in model_name:
    model = migraphx.parse_tf(model_name, is_nhwc=parser.nhwc)
else:
    raise Exception(
        "Unsupported file format. Supported formats are ONNX (.onnx) and Tensorflow Protobuf (.pb)"
    )

if args.fp16:
    migraphx.quantize_fp16(model, ins_names=["all"])
if args.int8:
    migraphx.quantize_int8(model,
                           target,
                           calibration=[],
                           ins_names=["dot", "convolution"])

target_name = args.target if args.target else "ref"
target = migraphx.get_target(target_name)
model.compile(target,
              offload_copy=(not args.offload_copy_off),
              fast_math=(not args.disable_fast_math))

if args.output_path:
    output_file = args.output_path
    if not os.path.dirname(output_file):
        output_file = "./saved_models/" + output_file
        os.makedirs("./saved_models", exist_ok=True)
    if not os.path.basename(output_file):
        output_file += os.path.basename(model_name)
else:
    output_file = "./saved_models/" + os.path.basename(model_name)
    os.makedirs("./saved_models", exist_ok=True)

if ".onnx" in output_file:
    output_file = output_file.replace(".onnx", "")
if ".pb" in output_file:
    output_file = output_file.replace(".pb", "")

if args.json:
    if ".msgpack" in output_file:
        output_file = output_file.replace(".msgpack", "")
    if ".json" not in output_file:
        output_file += ".json"
    migraphx.save(model, output_file, format="json")
else:
    if ".json" in output_file:
        output_file = output_file.replace(".json", "")
    if ".msgpack" not in output_file:
        output_file += ".msgpack"
    migraphx.save(model, output_file, format="msgpack")
