# MIGraphX `tools/`

This directory contains the helper scripts and supporting files used to develop, test,
profile, and maintain MIGraphX. They are not part of the runtime library; they are the
utilities that contributors and continuous integration run alongside it.

This page serves as an index for locating the appropriate tool. Tools that provide their
own detailed instructions link to a dedicated `README.md` within their subfolder.

> Most Python tools accept `-h`/`--help`; run `python3 tools/<tool>.py --help` for usage
> details. Unless otherwise noted, scripts should be run from the repository root.

## Task reference

| Task | Tool |
|---|---|
| Install build prerequisites or set up an environment | [`install_prereqs.sh`](#environment-and-build-setup), [`docker/`](#environment-and-build-setup) |
| Format source before pushing | [`format.py`](#code-quality-and-ci-gates) |
| Add or verify license-header year stamps | [`license_stamper.py`](#code-quality-and-ci-gates), [`check_stamped.py`](#code-quality-and-ci-gates) |
| Run the custom static-analysis (cppcheck) checks | [`cppcheck/`](#code-quality-and-ci-gates) |
| Regenerate the public C/Python API after editing headers | [`generate.py`](#api-and-code-generation) |
| Verify MIGraphX results against a reference | [`accuracy/`](#model-testing-and-verification), [`ort/`](#model-testing-and-verification) |
| Run a folder of ONNX test cases | [`test_runner.py`](#model-testing-and-verification), [`model_zoo/`](#model-testing-and-verification) |
| Convert an ONNX model's opset, or `.mxr` to ONNX | [`convert_onnx_version.py`](#model-testing-and-verification), [`converters/`](#model-testing-and-verification) |
| Determine the fastest environment-variable configuration for a model | [`autotune_perf/`](#performance-and-profiling) |
| Profile per-operator runtime or compile time | [`roctx.py`](#performance-and-profiling), [`compile_analysis.py`](#performance-and-profiling) |
| Build and test the ONNX Runtime and MIGraphX integration | [`build_and_test_onnxrt.sh`](#onnx-runtime-integration) |
| Review pull request status across the repository | [`pr_review_dashboard/`](#repository-and-workflow) |
| Enable syntax highlighting for MIGraphX IR dumps | [`syntax/`](#editor-support) |

---

## Environment and build setup

| Path | Description |
|---|---|
| `install_prereqs.sh` | Installs MIGraphX build prerequisites (CMake, the ROCm/MIOpen/rocBLAS development packages, `rbuild`, and Python dependencies) for Ubuntu and SLES. Used by the Docker images and for provisioning a development machine. Optional arguments: `install_prereqs.sh [PREFIX] [DEPS_DIR]`. |
| `requirements-py.txt` | Pinned Python packages (onnx, numpy, protobuf, pytest, and others) required by the Python-based tools and the ONNX Runtime unit tests. Install with `pip3 install -r tools/requirements-py.txt`. |
| `docker/` | Dockerfiles for reproducible build and test environments: `ubuntu_2204.dockerfile`, `ubuntu_2404.dockerfile`, `sles.docker`, `ort.dockerfile` (ONNX Runtime), `migraphx_with_onnxruntime_pytorch.docker`, and `therock_deb.docker`. |

## Code quality and CI gates

These scripts implement the `format`, `licensing`, and `cppcheck` gates that block pull
request merges.

| Path | Description |
|---|---|
| `format.py` | Runs `clang-format` on changed C/C++/HIP files and `yapf` on changed Python files. Run `python3 tools/format.py origin/develop` to display diffs, or add `-i` to apply changes in place. This should be run before every push. |
| `check_stamped.py` | Verifies that every file changed relative to a base branch carries the current-year MIT license stamp (the continuous integration `licensing` gate). Usage: `python3 tools/check_stamped.py origin/develop`. |
| `license_stamper.py` | Adds or updates license-header year stamps. `python3 tools/license_stamper.py origin/develop` stamps changed files; `--all` rewrites every tracked file using its last-commit year. |
| `stamp_status.py` | Shared helper for the two scripts above, providing the stamp-detection and year-update logic. Not intended to be run directly. |
| `git_tools.py` | Shared helper used by the format and stamp scripts to query git for the merge base, changed files, and commit year. Not intended to be run directly. |
| `cppcheck/` | MIGraphX's custom static analysis: a cppcheck addon (`migraphx.py`), XML pattern rules (`rules.xml`), and `test.sh` to run them. See [`cppcheck/test/README.md`](./cppcheck/test/README.md) for the test suite. |

## API and code generation

MIGraphX's stable C/C++/Python API and its type-erased interfaces are generated from
templates and header definitions. Edit the inputs described below, then regenerate.

| Path | Description |
|---|---|
| `generate.py` | Entry point for the CMake `generate` target. Regenerates `src/api/api.cpp`, `src/api/include/migraphx/migraphx.h`, and the type-erased headers under `src/include/migraphx/`, then applies clang-format. Run `make generate` from the build directory, or `python3 tools/generate.py -f <clang-format>`. |
| `api.py` | The API-generation engine consumed by `generate.py`. It parses the API definitions and emits the C header and `api.cpp`. |
| `te.py` | The type-erasure generator, which converts the interface headers in `include/` into their concrete type-erased classes. |
| `api/` | Input templates for the public API: `migraphx.h` and `api.cpp`, processed by `api.py`. |
| `include/` | Interface header definitions (`operation.hpp`, `pass.hpp`, `target.hpp`, `context.hpp`, and others) consumed by `te.py` to generate the type-erased versions. |
| `CMakeLists.txt` | Defines the `generate` custom target (requires Python 3 and clang-format). |

## Model testing and verification

| Path | Description |
|---|---|
| `accuracy/` | Compares MIGraphX output against ONNX Runtime for a given ONNX model and reports `PASSED` or `FAILED`. See [`accuracy/README.md`](./accuracy/README.md). |
| `ort/` | A lightweight ONNX Runtime driver for quickly running and validating an ONNX file through ONNX Runtime. See [`ort/README.md`](./ort/README.md). |
| `test_runner.py` | Runs a folder of ONNX test cases (a model with reference inputs and outputs) through MIGraphX on a chosen target (`ref` or `gpu`) with configurable tolerances. Usage: `python3 tools/test_runner.py <test_dir> --target gpu`. |
| `model_zoo/` | Helpers to fetch and run larger model collections: the ONNX Model Zoo and a dataset-driven test generator. See [`model_zoo/README.md`](./model_zoo/README.md), [`model_zoo/onnx_zoo/README.md`](./model_zoo/onnx_zoo/README.md), and [`model_zoo/test_generator/README.md`](./model_zoo/test_generator/README.md). |
| `convert_onnx_version.py` | Converts an ONNX model to a target opset, optionally inferring shapes. Usage: `python3 tools/convert_onnx_version.py --model in.onnx --output out.onnx --opset N`. |
| `converters/` | `mxr_to_onnx.py` reconstructs an ONNX graph from a serialized MIGraphX (`.mxr`) program, which is useful for inspecting or replaying compiled models. |

## Performance and profiling

| Path | Description |
|---|---|
| `autotune_perf/` | Sweeps MIGraphX environment-variable settings on top of `migraphx-driver perf`, reports the fastest configuration, and writes a sourceable `<model>.tune` file. See [`autotune_perf/README.md`](./autotune_perf/README.md). |
| `roctx.py` | Parses ROCTX marker traces (JSON) produced by a profiled MIGraphX run into per-operator CSV summaries for analysis. Usage: `python3 tools/roctx.py --json-path <trace.json> --out <dir>`. |
| `compile_analysis.py` | Parses a MIGraphX compile log for per-stage timings and renders an interactive HTML breakdown of compile time. Usage: `python3 tools/compile_analysis.py --file_path <log>`. |

## ONNX Runtime integration

Scripts used to build and continuously test the MIGraphX execution provider within ONNX
Runtime.

| Path | Description |
|---|---|
| `build_and_test_onnxrt.sh` | Builds ONNX Runtime with the MIGraphX execution provider and runs its test suites (used by the ONNX Runtime continuous integration image). |
| `pai_test_launcher.sh` | Runs `onnxruntime_test_all`, excluding the tests listed in `pai-excluded-tests.txt`. |
| `pai_provider_test_launcher.sh` | Runs `onnxruntime_provider_test`, excluding the tests listed in `pai-excluded-tests.txt`. |

## Repository and workflow

| Path | Description |
|---|---|
| `pr_review_dashboard/` | `pr-review-status.py` categorizes open pull requests by reviewer and approval state (terminal output or `--json`), accompanied by `pr-review-config.json` and an `index.html` dashboard view. Requires a `GITHUB_TOKEN`. |

## Editor support

| Path | Description |
|---|---|
| `syntax/` | `migx.vim` provides Vim syntax highlighting for MIGraphX IR (program and module) text dumps. |

---

### Conventions for tools

- New or modified `.py`, `.sh`, `.cpp`, `.hpp`, and similar files require a current-year MIT
  license header. Run `python3 tools/license_stamper.py origin/develop` to apply it.
- Format Python with `yapf` and C/C++ with `clang-format` via
  `python3 tools/format.py origin/develop -i`.
- Refer to the repository-root `README.md` and the `CONTRIBUTING`/`AGENTS.md` files for the
  complete build and pull request workflow.
