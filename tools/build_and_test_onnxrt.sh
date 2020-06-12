cd /onnxruntime
pip3 install -r requirements.txt
# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
export CXXFLAGS="-D__HIP_PLATFORM_HCC__=1"
./build.sh --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --test --use_migraphx
pip install /code/onnxruntime/build/Linux/Release/dist/*.whl
