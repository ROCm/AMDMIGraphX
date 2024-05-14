import numpy as np
from onnx.reference import ReferenceEvaluator

#X = np.array([[1.0,2.0,3.,4.], [2.,4.,5.,7.], [11.,13.,23.,17.], [2.,1.,31.,37.]], dtype=float)
X = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
             dtype=float)
#X = np.array([[0, 0, 0, 0], [0 ,0 ,0 ,0], [0, 0, 0, 0], [0 ,0 ,0 ,0]])

label_data = np.array([0, 3, 1, 2])
weights = np.array([1., 0.5, 2., 3.], dtype=float)

#sess = ReferenceEvaluator("softmaxcrossentropyloss_2d_no_reduction_weighted_test.onnx", verbose=1)
sess = ReferenceEvaluator("softmaxcrossentropyloss_2d_no_reduction_test.onnx",
                          verbose=1)
#results = sess.run(None, {"0": X, "1": label_data, "2":weights})
results = sess.run(None, {"0": X, "1": label_data})
print(results[0])

x = X  #np.array([1, 1], dtype=float)
print("Input")
print(x)
max_x = np.max(x, axis=1, keepdims=True)
print(max_x)
e_x = np.exp(x - max_x)
print(e_x)
sm = (e_x / np.sum(e_x, axis=1, keepdims=True))
print(sm)
logsm = np.log(sm)
print(-logsm)

print("Sum reduction")
print(np.sum(-logsm[0]))

print("mean reduction")
print(np.mean(-logsm))
