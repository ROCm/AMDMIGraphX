from argparse import ArgumentParser
from collections import namedtuple
import warnings

warnings.filterwarnings('ignore')

# datasets
from sample_generator.dataset.imagenet import ImageNet2012Val
from sample_generator.dataset.librispeech import LibriSpeechASR
from sample_generator.dataset.squad import SQuAD_HF

# models
from sample_generator.model.image import ResNet50_v1, ResNet50_v1_5, VitBasePatch16_224, TIMM_MobileNetv3_large
from sample_generator.model.text import BERT_large_uncased, DistilBERT_base_cased_distilled_SQuAD, RobertaBaseSquad2
from sample_generator.model.text import GPTJ, Llama2_7b_chat_hf, T5_base, Gemma_2b_it
from sample_generator.model.audio import Wav2Vec2_base_960h, WhisperSmallEn
from sample_generator.model.hybrid import ClipVitLargePatch14

# generator
from sample_generator.generator import generate_test_dataset

DatasetModelsPair = namedtuple('DatasetModelsPair', ['dataset', 'models'])

imagenet_models = (
    ResNet50_v1,
    ResNet50_v1_5,
    VitBasePatch16_224,
    TIMM_MobileNetv3_large,
    ClipVitLargePatch14,
)

squad_models = (
    BERT_large_uncased,
    DistilBERT_base_cased_distilled_SQuAD,
    RobertaBaseSquad2,
    GPTJ,
    T5_base,
    Gemma_2b_it,
    Llama2_7b_chat_hf,
)

librispeech_models = (Wav2Vec2_base_960h, WhisperSmallEn)

default_dataset_model_mapping = {
    "image": DatasetModelsPair(ImageNet2012Val, imagenet_models),
    "text": DatasetModelsPair(SQuAD_HF, squad_models),
    "audio": DatasetModelsPair(LibriSpeechASR, librispeech_models)
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        choices=['all', 'none'] + [model.name() for model in imagenet_models],
        nargs='+',
        default='all',
        dest='image_model_names',
        type=str,
        help=
        f'Image models to test with {ImageNet2012Val.name()} dataset samples')
    parser.add_argument(
        '--text',
        choices=['all', 'none'] + [model.name() for model in squad_models],
        nargs='+',
        default='all',
        dest='text_model_names',
        type=str,
        help=f'Text models to test with {SQuAD_HF.name()} dataset samples')
    parser.add_argument(
        '--audio',
        choices=['all', 'none'] +
        [model.name() for model in librispeech_models],
        nargs='+',
        default='all',
        dest='audio_model_names',
        type=str,
        help=
        f'Audio models to test with {LibriSpeechASR.name()} dataset samples')
    parser.add_argument(
        '--output-folder-prefix',
        default='generated',
        help='Output path will be "<this-prefix>/<dataset-name>/<model-name>"')
    parser.add_argument(
        '--sample-limit',
        default=5,
        help="Max number of samples generated. Use 0 to ignore it.")
    parser.add_argument(
        '--decode-limit',
        default=5,
        help=
        "Max number of sum-samples generated for decoder models. Use 0 to ignore it. (Only for decoder models)"
    )
    return parser.parse_args()


def get_dataset_models_pairs(dataset_type, model_names):
    if 'none' in model_names:
        return None
    ds_ms_mapping = default_dataset_model_mapping[dataset_type]
    if 'all' in model_names:
        return ds_ms_mapping

    return DatasetModelsPair(dataset=ds_ms_mapping.dataset,
                             models=(model for model in ds_ms_mapping.models
                                     if model.name() in model_names))


def main(image_model_names='all',
         text_model_names='all',
         audio_model_names='all',
         output_folder_prefix='generated',
         sample_limit=5,
         decode_limit=5):
    for dataset_type, model_names in zip(
        ('image', 'text', 'audio'),
        (image_model_names, text_model_names, audio_model_names)):
        dataset_model_pair = get_dataset_models_pairs(dataset_type,
                                                      model_names)
        if dataset_model_pair is None:
            print(f"Skip {dataset_type}...")
            continue

        dataset = dataset_model_pair.dataset
        for model in dataset_model_pair.models:
            generate_test_dataset(model(),
                                  dataset(),
                                  output_folder_prefix=output_folder_prefix,
                                  sample_limit=sample_limit,
                                  decode_limit=decode_limit)
    print(
        f'Use this to test MIGraphX with the result:\npython ../test_runner.py <generated_dataset_path>'
    )


if '__main__' == __name__:
    args = get_args()
    main(**vars(args))
