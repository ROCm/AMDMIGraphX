
# Not for release.  This test script is for develop/test only

import onnx
import onnxruntime as rt
# from https://onnxruntime.ai/docs/get-started/with-python.html
import numpy as np
print(" version: ", onnx.__version__, rt.__version__)


x = np.array(np.arange(2*2*4*3), dtype='f')
x = np.reshape(x, [2, 2, 4, 3])

y=np.ones([2, 2, 4, 7], dtype='f')

# rois=np.array([[0.1, 0.15, 0.6, 0.35],
#                 [0.1, 0.15, 2.6, 1.35]], dtype='f')

rois=np.array([
                [ 1.1, 0.73, 2.2, 1.13]], dtype='f')
sess = rt.InferenceSession('/workspace/AMDMIGraphX/test/onnx/roialign_half_pixel_test.onnx')
res = sess.run(['y'], {'x': x,
                    'rois': rois,
                    # 'batch_ind': [0, 1]})
                    'batch_ind': [0]})
print(res)
       
		
# model_file = "test/onnx/roialign_test.onnx"
# onnx_model = onnx.load(model_file)
# onnx.checker.check_model(onnx_model)


# #define the priority order for the execution providers
# EP_list = ['CPUExecutionProvider']

# aa = np.asarray(np.arange(3*2*4*5), dtype='f')
# # bi = np.reshape(aa, [3, 2, 4, 5])

# # initialize the model.onnx
# sess = rt.InferenceSession(model_file, providers=EP_list)
# x, rois, batch_ind = (np.reshape(aa, [3, 2, 4, 5]),
#             np.array([[0.1, 0.15, 0.6, 0.35],
#                       [2.1, 1.73, 3.8, 2.13]], dtype='f'),
#             np.array([0, 1], dtype='int64'))

# #  Use the parameter names defined in the onnx file
# output = sess.run(None, {'x':  x,
#                          'rois': rois,
#                          'batch_ind': batch_ind,
#                          })

# print(' output is ', output)


# # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
# output_name = sess.get_outputs()[0].name

# # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
# input_name = sess.get_inputs()[0].name
# print("Names are  ",input_name, output_name)

