# TensorRT-on-MIGraphX Samples

This directory contains the TensorRT-compatibility samples that exercise the
`commonapi` adaptation layer — an implementation of NVIDIA TensorRT's `nvinfer1`
interfaces on top of AMD MIGraphX.

```
examples/tensorrt/
├── CMakeLists.txt        <- standalone project (find HIP + MIGraphX/commonapi libs)
├── README.md             <- this file
├── samples/
│   ├── CMakeLists.txt
│   ├── common/           <- shared TensorRT sample helpers (logger, buffers, ...)
│   ├── utils/            <- shared utilities (file lock, cache, ...)
│   ├── sampleOnnxMNIST/
│   └── sampleCharRNN/
└── samples_data/         <- input data / weights
    ├── mnist/
    ├── char-rnn/
    └── resnet50/
```

| Sample            | What it does                                                                          |
|-------------------|---------------------------------------------------------------------------------------|
| `sampleOnnxMNIST` | Parses an MNIST ONNX model through the TensorRT ONNX parser and classifies a digit.   |
| `sampleCharRNN`   | Builds a character-level RNN (two LSTM layers) with the TensorRT API and runs inference. |

## Prerequisites

This is a **standalone** CMake project (configured out-of-source), separate from the
main MIGraphX build. Before building it you must have a **built** MIGraphX tree,
because the samples link the `commonapi` adaptation layer together with the MIGraphX
libraries it depends on:

- `libcommonapi.so`
- `libmigraphx.so`, `libmigraphx_c.so`, `libmigraphx_onnx.so`, `libmigraphx_ref.so`,
  `libmigraphx_gpu.so`

By default the project looks for these in the in-repo build tree at
`<repo-root>/build/lib`. If you built MIGraphX there (the usual
`cmake -S . -B build && cmake --build build` flow from the repo root), no extra
configuration is needed.

> Important: the samples also include MIGraphX headers from the source tree
> (`src/include`, `src/common_api/include`) and one internal impl header
> (`NetworkDefinition_impl.hpp`, used by `sampleCharRNN` for debug access). For ABI
> consistency the linked libraries must come from the **same** MIGraphX source tree
> as those headers — which is why the libraries default to the in-repo `build/lib`
> rather than a system install such as `/opt/rocm` (whose MIGraphX may be a different
> version and does not contain `commonapi`).

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

This produces:

```
build/samples/sampleOnnxMNIST/sampleOnnxMNIST
build/samples/sampleCharRNN/sampleCharRNN
```

If MIGraphX was built somewhere other than `<repo-root>/build`, point the project at
the directory that holds the shared libraries:

```bash
cmake -S . -B build -DMIGRAPHX_LIB_DIR=/path/to/migraphx/lib
```

## Run

The sample binaries link the MIGraphX/commonapi shared libraries by path, but it is
simplest to put that library directory on the loader path. Then pass the data
directory with `-d` / `--datadir` (data now lives under this folder's
`samples_data/`). The commands below are written relative to this directory.

```bash
export LD_LIBRARY_PATH=<repo-root>/build/lib:$LD_LIBRARY_PATH
```

### sampleOnnxMNIST

```bash
./build/samples/sampleOnnxMNIST/sampleOnnxMNIST --datadir ./samples_data/mnist/
```

Expected: an ASCII rendering of a digit and a classification, ending with:

```
&&&& PASSED TensorRT.sample_onnx_mnist ...
```

### sampleCharRNN

```bash
./build/samples/sampleCharRNN/sampleCharRNN --datadir ./samples_data/char-rnn/
```

Expected (example):

```
[I] RNN warmup sentence: Hi
[I] Expected output: ng of York,\n That thou hast so the
[I] Received: ng of York,\n That thou hast so the
&&&& PASSED TensorRT.sample_char_rnn ...
```

`sampleCharRNN` also has a built-in `--test` mode used during development to exercise
individual layer implementations (selected by the `TEST_NS` macro in
`sampleCharRNN.cpp`); it does not require `--datadir`:

```bash
./build/samples/sampleCharRNN/sampleCharRNN --test
```

## Notes

- The sample *sources* are unchanged by the relocation; only the `CMakeLists.txt`
  files were rewritten for a standalone build, and this `README.md` was added.
- Data directories are resolved at runtime via `--datadir`; no data path is compiled
  into the binaries.
