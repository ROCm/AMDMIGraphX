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
USE_LOCAL=1 ./test_models.sh /mnt/nas_share/onnx-model-zoo

# Or select a subset
USE_LOCAL=1 ./test_models.sh \
    /mnt/nas_share/onnx-model-zoo/text/machine_comprehension/t5 \
    /mnt/nas_share/onnx-model-zoo/vision/classification/shufflenet
```

## Results

Result are separated by dtype: `logs/fp32`, `logs/fp16` and `logs/int8`

> [!NOTE]
> `int8`/`qdq` models are already quantized in-graph, so they are only run in
> their native precision and logged under `logs/int8`; the fp16 pass is skipped
> for them.

### Helpers

```bash
# Something went wrong
grep -HRL PASSED logs
# Runtime error
grep -HRi RuntimeError logs/
# Accuracy issue
grep -HRl FAILED logs
```

## Cleanup

If at any point something fails, the following things might need cleanup:
- Remove `tmp_model` folder
- `git lfs prune` in `models`