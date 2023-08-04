#Install most recent stable version of pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Download ImageNet labels
curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

../../../tools/build_and_test_onnxrt.sh

pip3 install /onnxruntime/build/Release/Linux/dist/*.whl
