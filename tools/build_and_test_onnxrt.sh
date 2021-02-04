cd /onnxruntime
pip3 install -r requirements.txt
# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
export CXXFLAGS="-D__HIP_PLATFORM_HCC__=1 -w"
./build.sh --config Release --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --use_migraphx
/onnxruntime/build/Linux/Release/onnx_test_runner "/onnxruntime/cmake/external/onnx/onnx/backend/test/data/pytorch-converted" -c 1 -j1 -ep migraphx
/onnxruntime/build/Linux/Release/onnx_test_runner "/onnxruntime/cmake/external/onnx/onnx/backend/test/data/pytorch-operator"a -c 1 -j1 -ep migraphx
# pip3 install /code/onnxruntime/build/Linux/Release/dist/*.whl
