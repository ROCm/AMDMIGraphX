cd /onnxruntime
./build.sh --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --test --use_migraphx
pip install /code/onnxruntime/build/Linux/Release/dist/*.whl
