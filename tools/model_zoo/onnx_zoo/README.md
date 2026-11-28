# ONNX Zoo model tester

Helper script to test [`ONNX Zoo models`](https://onnx.ai/models/) which have test data with [`test_runner.py`](../../test_runner.py)

## Getting the repository

> [!IMPORTANT]
> Make sure to enable git-lfs.

```bash
git clone https://github.com/onnx/models.git --depth 1
```

## Running the tests

> [!IMPORTANT]
> The argument must point to a folder, not a file.

```bash
# VERBOSE=1 DEBUG=1 # use these for more log
# ATOL=0.001 RTOL=0.001 TARGET=gpu # are the default values
# FP16_ATOL=0.04 FP16_RTOL=0.04 # looser tolerances used for the fp16 pass
./test_models.sh models/validated
```

You can also pass multiple folders, e.g.:

```bash
./test_models.sh models/validated/text/machine_comprehension/t5/ models/validated/vision/classification/shufflenet/
```

## Running against pre-downloaded models

```bash
# Test every model found under the pre-downloaded location
USE_LOCAL=1 ./test_models.sh /datasets/onnx-model-zoo

# Test the curated CI set
USE_LOCAL=1 MODEL_LIST=/datasets/onnx-model-zoo/ci-models.txt \
    ./test_models.sh /datasets/onnx-model-zoo

# Or select a subset
USE_LOCAL=1 ./test_models.sh \
    /datasets/onnx-model-zoo/text/machine_comprehension/t5 \
    /datasets/onnx-model-zoo/vision/classification/shufflenet
```

## Model folders

Besides `.tar.gz` archives, a model can be a folder, which is tested in place:

```
<model folder>/
    <name>.onnx
    <name>.onnx.json         {} or {"dim_params": {"<dimension>": "<size>"}}
    test_data_set_<N>/
        cat.csv              a row per tensor: INPUT or OUTPUT, tensor name,
                             data file, ONNX type (tab separated, no header)
        <data file>.mgx      the tensor, as written by migraphx::save_argument
```

A folder is a model folder when its `.onnx` file has the `.onnx.json` or a
`test_data_set_*` folder beside it. Test sets without `cat.csv` are read from
`input_<i>.pb` and `output_<i>.pb`, as in the archives.

```bash
# Test every model folder below a root, grading each output by its RMS error
COMPARE=rms LOG_DIR=logs/atg ./test_models.sh /path/to/atg_zoo

# Or the listed ones; a line can end with tolerances for that model only
#   amd__yolov3
#   openai-community__gpt2/onnx RMS_TOL=0.002 FP16_RMS_TOL=0.1
COMPARE=rms MODEL_LIST=/path/to/atg_zoo/ci-models.txt ./test_models.sh /path/to/atg_zoo
```

- `COMPARE=allclose` (default) passes a case when every element is within
  `ATOL + RTOL * abs(expected)`.
- `COMPARE=rms` passes a case when the RMS error of every output is at most
  `RMS_TOL` (0.001), or `FP16_RMS_TOL` (0.08) in the fp16 pass. The RMS error is
  the one of `migraphx-driver verify`; the log has it for each output.
- A model without golden data is `SKIPPED` in accuracy and is left out of the
  health check. Performance still runs, with the `dim_params` of `.onnx.json`.
- A model folder with quantization operators in its graph is run once, like the
  `int8`/`qdq` archives.
- The logs of a model folder are named by its path below the root, with `/`
  replaced by `__`.

[`golden.py`](golden.py) reads the golden data and [`compare.py`](compare.py)
grades the outputs; both can be imported by other tools.

## Results

Accuracy and performance logs are written under `logs/accuracy` and `logs/perf`;
set `LOG_DIR` to write them elsewhere. FP16 runs use an `_fp16` filename suffix.
`summarize.py <logs> --title <title>` writes the summary `README.md` there.

> [!NOTE]
> `int8`/`qdq` models are already quantized in-graph, so they are only run in
> their native precision; the fp16 pass is skipped for them.

### Helpers

```bash
# Something went wrong
grep -HRL PASSED logs
# Runtime error
grep -HRi RuntimeError logs/
# Accuracy issue
grep -HRl FAILED logs/accuracy
```

## Cleanup

If at any point something fails, the following things might need cleanup:
- Remove `tmp_model` folder
- `git lfs prune` in `models`