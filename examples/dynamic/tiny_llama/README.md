# Dynamic Tiny-Llama

This example downloads a small Llama decoder, compiles prompt lengths 1 through
64 into one symbolic MIGraphX program, verifies three representative lengths,
and measures length 17.

The model is the legacy no-past-input decoder from
[`optimum-internal-testing/tiny-random-llama`](https://huggingface.co/optimum-internal-testing/tiny-random-llama/tree/merged-onnx/onnx_legacy).
It accepts `input_ids` and `attention_mask`, then returns logits and the
generated key/value cache. The preparation step fixes the model's batch
metadata to 1 while preserving its shared `sequence_length` symbol.

## Prepare the model

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 prepare.py
```

This writes `models/tiny_llama/decoder_model.onnx`.

## Compile, verify, and measure

Make `migraphx-driver` available on `PATH`, or set `MIGRAPHX_DRIVER` to its
absolute path, then run:

```bash
bash run.sh
```

The script creates `tiny_llama_dynamic.mxr`. The same compiled program is
verified at sequence lengths 1, 17, and 64. It is then measured at sequence
length 17 for 100 iterations.

Custom model and output paths can be supplied as positional arguments:

```bash
bash run.sh /path/to/decoder_model.onnx /path/to/tiny_llama_dynamic.mxr
```
