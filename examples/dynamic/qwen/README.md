# Dynamic Qwen2

This example exports the public 4.9-million-parameter
[tiny-random Qwen2](https://huggingface.co/onnx-internal-testing/tiny-random-Qwen2ForCausalLM)
model to a cache-free ONNX graph. It compiles input lengths 1 through 64 into
one symbolic MIGraphX program, verifies three representative lengths, and
measures length 17.

The exported decoder keeps Qwen2's attention, rotary-position, RMS
normalization, and gated MLP computations while omitting key/value caching.
`input_ids` and `position_ids` have shape `[1, sequence_length]`. The additive
`attention_mask` has shape
`[1, 1, sequence_length, sequence_length]`.

## Prepare the model

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 prepare.py
```

This writes `models/qwen/model.onnx`.

## Compile, verify, and measure

Make `migraphx-driver` available on `PATH`, or set `MIGRAPHX_DRIVER` to its
absolute path, then run:

```bash
bash run.sh
```

The script creates `qwen_dynamic.mxr`. The same compiled program is verified at
sequence lengths 1, 17, and 64. It is then measured at sequence length 17 for
100 iterations.

Custom model and output paths can be supplied as positional arguments:

```bash
bash run.sh /path/to/model.onnx /path/to/qwen_dynamic.mxr
```
