cd /onnxruntime
pip install -r requirements.txt
# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
./build.sh --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --test --use_migraphx
pip install /code/onnxruntime/build/Linux/Release/dist/*.whl
