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
"""Read the golden inputs and outputs stored beside a zoo model.

load_cases() is the entry point. A model folder holds one test case per
test_data_set_*/ folder, in one of these formats:

    cat.csv       a tab-separated row per tensor: INPUT or OUTPUT, the tensor
                  name, a data file name and the ONNX type. The tensor is read
                  from <data file>.mgx (migraphx::save_argument).
    *.pb          input_<i>.pb and output_<i>.pb TensorProto files.

Older archives hold test_data_*.npz files instead of folders. To support
another format add a reader beside read_cat_case() and select it in
read_case().
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
import test_runner as tr  # noqa: E402
import migraphx  # noqa: E402


def read_cat_case(folder, params, outs):
    tensors = {'INPUT': {}, 'OUTPUT': {}}
    with open(os.path.join(folder, 'cat.csv')) as cat:
        for line in cat.read().splitlines():
            if not line.strip():
                continue
            fields = line.split('\t')
            if len(fields) < 3 or fields[0] not in tensors:
                raise ValueError("{}: bad cat.csv row: {!r}".format(
                    folder, line))
            kind, name, data_file = fields[:3]
            tensors[kind][name] = np.array(
                migraphx.argument.load(os.path.join(folder,
                                                    data_file + '.mgx')))

    missing = [name for name in params if name not in tensors['INPUT']]
    missing += [name for name in outs if name not in tensors['OUTPUT']]
    if missing:
        raise ValueError("{}: cat.csv has no row for: {}".format(
            folder, ', '.join(missing)))

    return ({name: tensors['INPUT'][name]
             for name in params}, [tensors['OUTPUT'][name] for name in outs])


def read_pb_case(folder, params, outs):
    return tr.wrapup_inputs(folder, params), tr.read_outputs(folder, outs)


def read_npz_case(path, params):
    inputs, gold = tr.load_npz_case(path)
    return tr.map_inputs(params, inputs), gold


def read_case(folder, params, outs):
    """({input name: array}, [golden outputs in model output order]) of one
    test_data_set_*/ folder."""
    if os.path.isfile(os.path.join(folder, 'cat.csv')):
        return read_cat_case(folder, params, outs)
    return read_pb_case(folder, params, outs)


def load_cases(folder, params=None, outs=None):
    """(input_shapes, [(label, inputs, gold)]) for the model in folder, or
    (None, []) when it has no golden data. params and outs are the model's
    input and output names; they are read from the model when omitted."""
    if params is None or outs is None:
        model_path = os.path.join(folder, tr.get_model_name(folder))
        params = tr.model_parameter_names(model_path)
        outs = tr.model_output_names(model_path)

    cases = [(name, ) + read_case(os.path.join(folder, name), params, outs)
             for name in sorted(os.listdir(folder))
             if name.startswith('test_data_set_')
             and os.path.isdir(os.path.join(folder, name))]
    if not cases:
        for i, path in enumerate(
                sorted(glob.glob(os.path.join(folder, 'test_data_*.npz')))):
            cases.append(('test_data_{}'.format(i), ) +
                         read_npz_case(path, params))
    if not cases:
        return None, []

    return {name: data.shape for name, data in cases[0][1].items()}, cases


def read_dim_params(model_path):
    """{name: size} from the dim_params of <model>.onnx.json, which bind the
    model's symbolic dimensions. Empty when the file is absent."""
    path = model_path + '.json'
    if not os.path.isfile(path):
        return {}
    with open(path) as config:
        dim_params = json.load(config).get('dim_params', {})

    return {name: int(size) for name, size in dim_params.items()}
