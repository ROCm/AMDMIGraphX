from argparse import ArgumentParser
import numpy as np
import pickle
from pathlib import Path
import os
import evaluate
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

G_MAX_TOK_LEN = 1024
G_LLAMA2_EOS = 2
SAMPLE_SIZE = 10

DATASET_PATH = "/dataset/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
RESULT_PATH = "build/results.txt"

def main(dataset_path, result_path, sample_size, sequence_size):
    tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            model_max_length=sequence_size,
            padding_side="left",
            use_fast=False,)

    metric = evaluate.load("rouge")
    nltk.download("punkt")

    _p = Path(DATASET_PATH)
    if _p.exists():
        with _p.open(mode="rb") as f:
            d = pickle.load(f)


    target = d['output'].to_list()
    targets = target[0:sample_size]
    results, gen_tok_len = readResult(result_path)

    preds = tokenizer.batch_decode(
            results, skip_special_tokens=True
        )

    postprocess_text(preds, target)

    result = metric.compute(
            predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
        )

    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    result = {
            **result,
            "gen_len": np.sum(prediction_lens),
            "gen_num": gen_num,
            "gen_tok_len": gen_tok_len,
            "tokens_per_sample": round(gen_tok_len / gen_num, 1),
        }

    print("\nResults\n")
    print(result)

def readResult(path):
    results = []
    tok_len = 0
    f = open(path, "r")
    for res in f:
        result = res.split(",")
        result = [int(num_res) for num_res in result]
        results.append(result)
        tok_len += len(result)
    return results, tok_len

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="Path to the dataset pickle file",
        default=DATASET_PATH
    )

    parser.add_argument(
        "-r",
        "--result_path",
        help="Path to output tokens result file",
        default=RESULT_PATH
    )

    parser.add_argument(
        "-size",
        "--sample-size",
        help="Sample size of dataset",
        type=int,
        default=SAMPLE_SIZE
    )

    parser.add_argument(
        "-seq_size",
        "--sequence_size",
        help="Size of sequence",
        type=int,
        default=G_MAX_TOK_LEN
    )

    main(**vars(parser.parse_args()))