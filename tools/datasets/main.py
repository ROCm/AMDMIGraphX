from dataset import ImageNet2012Val, SQuAD_HF, LibriSpeechASR
from model import ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large  #for imagemet
from model import DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2  # for squad
from model import Wav2Vec2_base_960h, WhisperSmallEn  # for librispeech
from generator import generate_test_dataset

model_dataset_pairs = [{
    "dataset":
    ImageNet2012Val,
    "models":
    [ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large]
}, {
    "dataset":
    SQuAD_HF,
    "models": [DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2]
}, {
    "dataset": LibriSpeechASR,
    "models": [Wav2Vec2_base_960h, WhisperSmallEn]
}]

for model_dataset_map in model_dataset_pairs:
    dataset = model_dataset_map["dataset"]
    for model in model_dataset_map["models"]:
        generate_test_dataset(model(),
                              dataset(),
                              sample_limit=5,
                              decode_limit=5)
print(
    f"Use this to test migraphx with the result:\npython ../test_runner.py <generated_dataset_path>"
)
