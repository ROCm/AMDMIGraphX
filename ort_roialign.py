# Not for release.  This test script is for develop/test only

import onnx
import onnxruntime as rt
# from https://onnxruntime.ai/docs/get-started/with-python.html
import numpy as np
print(" version: ", onnx.__version__, rt.__version__)

x = np.array(np.arange(10 * 5 * 4 * 7), dtype='f')
x = np.reshape(x, [10, 5, 4, 7])

y = np.ones([10, 5, 4, 7], dtype='f')

rois = np.array([[0.1, 0.15, 0.6, 0.35], [2.1, 1.73, 3.8, 2.13]], dtype='f')

themodel = 'roialign_test.onnx'
sess = rt.InferenceSession('/workspace/AMDMIGraphX/test/onnx/' + themodel)
res = sess.run(['y'], {'x': x, 'rois': rois, 'batch_ind': [1, 0]})

print(' ORT test model is ' + themodel + ', rois_data is \n', rois,
      ' result is \n', res)
