import toml
import torch
import sys


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


mlcommons_inference_path = './inference/'  # specify relative path for MLCommons inference
checkpoint_path = 'rnnt.pt'
config_toml = 'inference/retired_benchmarks/speech_recognition/rnnt/pytorch/configs/rnnt.toml'
config = toml.load(config_toml)
rnnt_vocab = config['labels']['labels']
sys.path.insert(
    0, mlcommons_inference_path +
    'retired_benchmarks/speech_recognition/rnnt/pytorch')

from model_separable_rnnt import RNNT


# model = RNNT(config['rnnt'], len(rnnt_vocab) + 1, feature_config=config['input_eval'])
def add_blank_label(labels):
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of symbols")
    labels.append("<BLANK>")
    return labels


dataset_vocab = config['labels']['labels']
rnnt_vocab = add_blank_label(dataset_vocab)
featurizer_config = config['input_eval']
model = RNNT(feature_config=featurizer_config,
             rnnt=config['rnnt'],
             num_classes=len(rnnt_vocab))

model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path))
model.to('cuda')
model.eval()

seq_length, batch_size, feature_length = 133, 1, 240
inp = torch.randn([seq_length, batch_size, feature_length]).to('cuda')
feature_length = torch.LongTensor([seq_length]).to('cuda')
x_padded, x_lens = model.encoder(inp, feature_length)
torch.onnx.export(model.encoder, (inp, feature_length),
                  "rnnt_encoder.onnx",
                  opset_version=12,
                  input_names=['input', 'feature_length'],
                  output_names=['x_padded', 'x_lens'],
                  dynamic_axes={'input': {
                      0: 'seq_len',
                      1: 'batch'
                  }})

symbol = torch.LongTensor([[20]]).to('cuda')
hidden = torch.randn([2, batch_size,
                      320]).to('cuda'), torch.randn([2, batch_size,
                                                     320]).to('cuda')
g, hidden = model.prediction.forward(symbol, hidden)
torch.onnx.export(model.prediction, (symbol, hidden),
                  "rnnt_prediction.onnx",
                  opset_version=12,
                  input_names=['symbol', 'hidden_in_1', 'hidden_in_2'],
                  output_names=['g', 'hidden_out_1', 'hidden_out_2'],
                  dynamic_axes={
                      'symbol': {
                          0: 'batch'
                      },
                      'hidden_in_1': {
                          1: 'batch'
                      },
                      'hidden_in_2': {
                          1: 'batch'
                      }
                  })

f = torch.randn([batch_size, 1, 1024]).to('cuda')
model.joint.forward(f, g).to('cuda')
torch.onnx.export(model.joint, (f, g),
                  "rnnt_joint.onnx",
                  opset_version=12,
                  input_names=['0', '1'],
                  output_names=['result'],
                  dynamic_axes={
                      '0': {
                          0: 'batch'
                      },
                      '1': {
                          0: 'batch'
                      }
                  })
