# datasets
from sample_generator.dataset.imagenet import ImageNet2012Val
from sample_generator.dataset.librispeech import LibriSpeechASR
from sample_generator.dataset.squad import SQuAD_HF

# models
from sample_generator.model.image import ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large  #for imagenet
from sample_generator.model.text import DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2, GPTJ  # for squad
from sample_generator.model.audio import Wav2Vec2_base_960h, WhisperSmallEn  # for librispeech
from sample_generator.generator import generate_test_dataset

model_dataset_pairs = [{
    "dataset":
    ImageNet2012Val,
    "models":
    [ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large]
}, {
    "dataset":
    SQuAD_HF,
    "models": [DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2, GPTJ]
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
