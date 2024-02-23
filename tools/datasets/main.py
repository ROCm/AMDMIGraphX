from dataset import ImageNet2012Val
from model import ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large
from generator import generate_test_dataset

model_dataset_pairs = [{
    "dataset":
    ImageNet2012Val,
    "models":
    [ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large]
}]

for model_dataset_map in model_dataset_pairs:
    dataset = model_dataset_map["dataset"]
    for model in model_dataset_map["models"]:
        generate_test_dataset(model(), dataset(), limit=5)
print(
    f"Use this to test migraphx with the result:\npython ../test_runner.py <generated_dataset_path>"
)
