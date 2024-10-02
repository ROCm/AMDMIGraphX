
# Not for release.  This test script is for develop/test only

import onnx
import onnxruntime as rt
# from https://onnxruntime.ai/docs/get-started/with-python.html
import numpy as np
print(" version: ", onnx.__version__, rt.__version__)


x = np.array(np.arange(2*2*4*3), dtype='f')
x = np.reshape(x, [2, 2, 4, 3])

y=np.ones([2, 2, 4, 3], dtype='f')

# matches roialign_half_pixel_verify_test
# rois=np.array([[0.1, 0.15, 0.6, 0.35],
#                 [1.1, 0.73, 1.9, 1.13]], dtype='f')
# matches roialign_half_pixel_oob_verify_test
rois=np.array([
                [1.1, 0.73, 1.7, 1.13],
                [1.1, 0.73, 2.6, 1.13]
                #         [1.1, 0.73, 2.6, 1.13]
                ], dtype='f')

# rois=np.array([
#                 [ 1.1, 0.73, 2.2, 1.13]], dtype='f')
sess = rt.InferenceSession('/workspace/AMDMIGraphX/test/onnx/roialign_half_pixel_test.onnx')
# sess = rt.InferenceSession('/workspace/AMDMIGraphX/test/onnx/roialign_one_roi_asdf_test.onnx') 
res = sess.run(['y'], {'x': x,
                    'rois': rois,
                    'batch_ind': [0, 1]})
                  #   'batch_ind': [0]})
print(' ORT test model is roialign_one_roi_asdf_test.onnx, rois_data is \n',rois, 
      ' result is \n', res)
       
		