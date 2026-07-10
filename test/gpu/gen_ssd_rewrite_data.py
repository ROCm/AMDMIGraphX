#!/usr/bin/env python3
#
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
"""
Generate the boxes/scores inputs and the gold TopK values for the SSD-style
NonMaxSuppression -> Gather -> TopK chain exercised by test/gpu/ssd_rewrite.cpp
(TEST_CASE test_ssd_rewrite).

MIGraphX keeps the NMS output static and zero-padded up to max_boxes, whereas
onnxruntime trims it to num_selected before the Gather/TopK. The ORT result is
therefore the correct, unpadded answer that the MIGraphX where-mask rewrite
(find_nms_gather_topk in src/simplify_dyn_ops.cpp) must reproduce. This prints
C++ initializers to paste into the test.

Requires: onnx, onnxruntime, numpy.
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def build_model(n, k, iou, score_thr, center_point_box):
    """NonMaxSuppression -> Slice(box col) -> Squeeze -> Gather(scores) -> TopK."""
    boxes  = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, n, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, n])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [k])

    initializers = [
        helper.make_tensor("max_out", TensorProto.INT64, [1], [n]),
        helper.make_tensor("iou", TensorProto.FLOAT, [1], [float(iou)]),
        helper.make_tensor("score_thr", TensorProto.FLOAT, [1], [float(score_thr)]),
        helper.make_tensor("col_starts", TensorProto.INT64, [1], [2]),
        helper.make_tensor("col_ends", TensorProto.INT64, [1], [3]),
        helper.make_tensor("col_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("k", TensorProto.INT64, [1], [k]),
    ]

    nodes = [
        helper.make_node("NonMaxSuppression",
                         ["boxes", "scores", "max_out", "iou", "score_thr"],
                         ["selected"], center_point_box=center_point_box),
        helper.make_node("Slice", ["selected", "col_starts", "col_ends", "col_axes"], ["box_col"]),
        helper.make_node("Squeeze", ["box_col", "col_axes"], ["box_col_1d"]),
        helper.make_node("Reshape", ["scores", "flat_shape"], ["scores_flat"]),
        helper.make_node("Gather", ["scores_flat", "box_col_1d"], ["gathered"], axis=0),
        helper.make_node("TopK", ["gathered", "k"], ["values", "topk_idx"],
                         axis=0, largest=1, sorted=1),
    ]

    graph = helper.make_graph(nodes, "ssd_rewrite", [boxes, scores], [values],
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def run_nms_only(boxes, scores, n, iou, score_thr, center_point_box):
    """Report the trimmed selection so the C++ gold can be sanity checked."""
    b = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, n, 4])
    s = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, n])
    out = helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3])
    node = helper.make_node("NonMaxSuppression",
                            ["boxes", "scores", "max_out", "iou", "score_thr"],
                            ["selected"], center_point_box=center_point_box)
    initializers = [
        helper.make_tensor("max_out", TensorProto.INT64, [1], [n]),
        helper.make_tensor("iou", TensorProto.FLOAT, [1], [float(iou)]),
        helper.make_tensor("score_thr", TensorProto.FLOAT, [1], [float(score_thr)]),
    ]
    graph = helper.make_graph([node], "nms", [b, s], [out], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"boxes": boxes, "scores": scores})[0]


def fmt_rows(arr, per_row):
    flat = arr.flatten().tolist()
    lines = []
    for i in range(0, len(flat), per_row):
        lines.append("    " + ", ".join(f"{v:.4f}f" for v in flat[i:i + per_row]) + ",")
    return "\n".join(lines)


def main():
    n, k = 4, 3
    iou, score_thr = 0.5, 0.0
    center_point_box = 0

    # Corner format [y1, x1, y2, x2]. box 3 duplicates box 2, so NMS suppresses it
    # (num_selected = 3 < max_boxes = 4). box 0 has the highest score, so the padded
    # row (which resolves to box 0) would corrupt the TopK without the where-mask.
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 2.0, 1.0, 3.0],
        [0.0, 4.0, 1.0, 5.0],
        [0.0, 4.0, 1.0, 5.0],
    ]], dtype=np.float32)
    scores = np.array([[[0.9, 0.8, 0.7, 0.6]]], dtype=np.float32)

    model = build_model(n, k, iou, score_thr, center_point_box)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    values = sess.run(None, {"boxes": boxes, "scores": scores})[0]

    selected = run_nms_only(boxes, scores, n, iou, score_thr, center_point_box)

    print("// === paste into test/gpu/ssd_rewrite.cpp (TEST_CASE test_ssd_rewrite) ===")
    print("// Values generated from onnxruntime CPU EP")
    print(f"std::vector<float> boxes_vec = {{\n{fmt_rows(boxes, 4)}}};")
    print("std::vector<float> scores_vec = {" +
          ", ".join(f"{v:.4f}f" for v in scores.flatten()) + "};")
    print("std::vector<float> gold = {" +
          ", ".join(f"{v:.4f}f" for v in values.flatten()) + "};")
    print(f"// k = {k}, num_selected = {selected.shape[0]}")
    print(f"// selected box indices: {selected[:, 2].tolist()}")


if __name__ == "__main__":
    main()
