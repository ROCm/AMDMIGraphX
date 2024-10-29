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
import migraphx

import onnx
import onnxruntime as rt
import numpy as np


def test_roialign():

    data = np.array(np.arange(2*2*4*3), dtype='f')
    data = np.reshape(data, [2, 2, 4, 3])
    roi_data = np.array([[ 0.1, 0.15, 0.6, 0.35], [ 0.1, 1.73, 0.8, 1.13]], dtype='f')
    index_data = np.array([1, 0], dtype='int64')

    # Create a program
    p = migraphx.program()
    mm = p.get_main_module()
    x = mm.add_literal(data)
    roi = mm.add_literal(roi_data)
    roi_index = mm.add_literal(index_data)
    
    roialign_op = mm.add_instruction(
        migraphx.op('roialign', coordinate_transformation_mode='half_pixel',
                    output_height=3,
                    output_width=2,
                    spatial_scale = 0.9,
                    sampling_ratio = 2),
                    [x, roi, roi_index])
    
    mm.add_return([roialign_op])
    p.compile(migraphx.get_target("ref"))
    params = {}

    mgx_result = p.run(params)[-1]

    # Make an ORT session from the *.onnx file
    themodel = 'roialign_half_pixel_test.onnx'
    sess = rt.InferenceSession('../onnx/' + themodel)
    
    
    #     TODO:  this output warning appears important because Migraphx and ORT results have 
    #   different shapes:  (2, 2, 2, 3) vs (2, 2, 3, 2)
    #   But is the Migraphx result an error?

    # root@rocm-rome-6:/workspace/AMDMIGraphX/test/py# python3 ./test_roialign.py 
    # 2024-10-28 22:13:54.675372471 [W:onnxruntime:, graph.cc:109 MergeShapeInfo] Error merging shape info for output. 'y' source:{2,2,3,2} target:{3,2,2,2}. Falling back to lenient merge.
    
    res = sess.run(['y'], {'x': data, 'rois': roi_data, 'batch_ind': index_data})
    print(' mgx result is a list with 1 item of shape ', mgx_result.get_shape().lens(), mgx_result)
    print(' ORT result is a list with 1 item of shape', res[0].shape, res)
    assert np.allclose(mgx_result, res[-1], rtol=1e-05, atol=1e-08, equal_nan=False)


if __name__ == "__main__":
    test_roialign()



