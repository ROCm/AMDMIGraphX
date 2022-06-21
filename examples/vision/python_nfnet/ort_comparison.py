import numpy
import onnxruntime as rt

sess = rt.InferenceSession("dm_nfnet_f0.onnx")

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

x = numpy.random.random((1, 3, 192, 192))
x = x.astype(numpy.float32)

import migraphx
model = migraphx.parse_onnx("dm_nfnet_f0.onnx")
model.compile(migraphx.get_target("gpu"))
print(model.get_parameter_names())
print(model.get_parameter_shapes())
print(model.get_output_shapes())

result_migraphx = model.run({"inputs": x})
result_ort = sess.run([output_name], {input_name: x})

result_migraphx = result_migraphx[0].tolist()

for i in range(10):
    x = numpy.random.random((1, 3, 192, 192))
    x = x.astype(numpy.float32)

    result_migraphx = model.run({"inputs": x})
    result_ort = sess.run([output_name], {input_name: x})

    try:
        numpy.testing.assert_allclose(result_migraphx[0].tolist(),
                                      result_ort[0][0],
                                      rtol=1e-02)
        print(f"Test #{i} completed.")
    except AssertionError as e:
        print(e)
