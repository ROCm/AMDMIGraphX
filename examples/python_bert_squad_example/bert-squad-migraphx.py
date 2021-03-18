import numpy as np
import json
import time
import os.path
from os import path
import sys

import tokenization
from run_onnx_squad import *

print(sys.path)

migx_lib_path = "/opt/rocm/lib"
if migx_lib_path not in sys.path:
    sys.path.append(migx_lib_path)


import migraphx

jsonData = {
    "version":
    "1.4",
    "data": [{
        "paragraphs": [{
            "context":
            "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
            "qas": [{
                "question": "where is the businesses choosing to go?",
                "id": "1"
            }]
        }],
        "title":
        "Conference Center"
    }]
}
#######################################3
#TODO Download uncased files and stuff

#######################################
input_file = 'inputs.json'
with open(input_file) as json_file:
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))

# preprocess input
predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=True)

my_list = []

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
    eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length)

#######################################
# Compile
print("INFO: Parsing and compiling the model...")
model = migraphx.parse_onnx("bertsquad-10.onnx")
model.compile(migraphx.get_target("gpu"))
model.print()

model.get_parameter_names()
model.get_parameter_shapes()

os._exit(0)

n = len(input_ids)
bs = batch_size
all_results = []
start = timer()

for idx in range(0, n):
    item = eval_examples[idx]
    print("item")
    print(item)
    
    # this is using batch_size=1
    # feed the input data as int64
    #data = {
    #    "unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64),
    #    "input_ids:0": input_ids[idx:idx + bs],
    #    "input_mask:0": input_mask[idx:idx + bs],
    #    "segment_ids:0": segment_ids[idx:idx + bs]
    #}

    #result = session.run(["unique_ids:0", "unstack:0", "unstack:1"], data)
    #results = model.run({'data': data}) # example
    result = model.run({
        "unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64),
        "input_ids:0": input_ids[idx:idx + bs],
        "input_mask:0": input_mask[idx:idx + bs],
        "segment_ids:0": segment_ids[idx:idx + bs]
    })
    print(result)
    print(result[0])
    print(result[1])

    ##in_batch = result[0].shape[0]
    in_batch = len(result[0].tolist())
    print("in_batch")
    print(in_batch)
    start_logits = [float(x) for x in result[0].tolist()]
    end_logits = [float(x) for x in result[1].tolist()]
    print(start_logits)
    print(end_logits)
    for i in range(0, in_batch):
        unique_id = len(all_results)
        all_results.append(
            RawResult(unique_id=unique_id,
                      start_logits=start_logits,
                      end_logits=end_logits))
        print("RawResult")
        print(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))


output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
write_predictions(eval_examples, extra_data, all_results,
                n_best_size, max_answer_length,
                True, output_prediction_file, output_nbest_file)

import json
with open(output_prediction_file) as json_file:  
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))