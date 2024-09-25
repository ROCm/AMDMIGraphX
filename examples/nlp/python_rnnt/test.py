import migraphx as mgx
import os

onnx_file = 'models/rnnt//rnnt_encoder/model.onnx'
shapes = {'input': [196, 1, 240], 'feature_length': [1]}
model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=False,
                          offload_copy=False)
mxr_file = 'models/rnnt//rnnt_encoder/model_fp32_gpu.mxr'
print(f"Saving model to {mxr_file}")
os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
mgx.save(model, mxr_file, format="msgpack")


onnx_file = 'models/rnnt//rnnt_prediction/model.onnx'
shapes = {
                "symbol": [1, 1],
                "hidden_in_1": [2, 1, 320],
                "hidden_in_2": [2, 1, 320]
            }
model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=False,
                          offload_copy=False)
mxr_file = 'models/rnnt//rnnt_prediction/model_fp32_gpu.mxr'
print(f"Saving model to {mxr_file}")
os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
mgx.save(model, mxr_file, format="msgpack")

onnx_file = 'models/rnnt//rnnt_joint/model.onnx'
shapes = {
                "0": [1, 1, 1024],
                "1": [1, 1, 320]
            }
model = mgx.parse_onnx(onnx_file, map_input_dims=shapes)
model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=False,
                          offload_copy=False)
mxr_file = 'models/rnnt//rnnt_joint/model_fp32_gpu.mxr'
print(f"Saving model to {mxr_file}")
os.makedirs(os.path.dirname(mxr_file), exist_ok=True)
mgx.save(model, mxr_file, format="msgpack")