import numpy as np
import json
import time
import os.path
from os import path
import sys

import tokenizers
from run_onnx_squad import *

import migraphx

#######################################
input_file = 'inputs_amd.json'
with open(input_file) as json_file:
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))

# preprocess input
predict_file = 'inputs_amd.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenizers.BertWordPieceTokenizer(vocab_file)

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
    eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length)

#######################################
# Compile
print("INFO: Parsing and compiling the model...")
model = migraphx.parse_onnx("bertsquad-10.onnx")
model.compile(migraphx.get_target("gpu"))
#model.print()

print(model.get_parameter_names())
print(model.get_parameter_shapes())

n = len(input_ids)
bs = batch_size
all_results = []

for idx in range(0, n):
    item = eval_examples[idx]
    print(item)

    result = model.run({
        "unique_ids_raw_output___9:0":
        np.array([item.qas_id], dtype=np.int64),
        "input_ids:0":
        input_ids[idx:idx + bs],
        "input_mask:0":
        input_mask[idx:idx + bs],
        "segment_ids:0":
        segment_ids[idx:idx + bs]
    })

    in_batch = result[1].get_shape().lens()[0]
    start_logits = [float(x) for x in result[1].tolist()]
    end_logits = [float(x) for x in result[0].tolist()]
    for i in range(0, in_batch):
        unique_id = len(all_results)
        all_results.append(
            RawResult(unique_id=unique_id,
                      start_logits=start_logits,
                      end_logits=end_logits))

output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
write_predictions(eval_examples, extra_data, all_results, n_best_size,
                  max_answer_length, True, output_prediction_file,
                  output_nbest_file)

with open(output_prediction_file) as json_file:
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))
