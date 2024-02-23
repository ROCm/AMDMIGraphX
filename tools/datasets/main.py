from dataset import get_imagenet_dataset_iterator, imagenet_transform
from model import download_resnet_model, RESNET_MODEL_NAME
from generator import generate_test_dataset

resnet_output_path = "imagenet/resnet50/"
model_path = f"imagenet/resnet50/{RESNET_MODEL_NAME}"

download_resnet_model(model_path)
generate_test_dataset(model_path, get_imagenet_dataset_iterator(),
                      imagenet_transform, 5)
print(
    f"Use this to test migraphx with the result:\npython ../test_runner.py {resnet_output_path}"
)
