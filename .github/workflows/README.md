# Workflows

## `add-to-project.yaml`

<p>
This workflow adds pull requests and issues to a specific GitHub project board when they are opened.
</p>

- ## Trigger
The workflow is triggered by the following events:

> - A pull request being opened.

> - An issue being opened.

- ## Jobs
The workflow has a single job named `add-to-project`. The following step is executed in this job:
> - The `add-to-project` job uses the `actions/add-to-project@v0.4.0` action to add pull requests and issues to a specific project board. The `with` parameters are `project-url` and `github-token`, which specify the URL of the project board and the GitHub token used to authenticate the action.

For more details, please refer to the [add-to-project.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/add-to-project.yaml) file in the repository.

---
## `benchmark.yaml`

<p>
This workflow runs the MiGraphX performance benchmarks and generates comparison reports by comparing the results with the reference data.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [benchmarks.yml (archived)](https://github.com/ROCmSoftwarePlatform/actions/blob/main/.github/workflows/benchmarks.yml)

- ## Input Parameters
The workflow uses the following input parameters:

> - `rocm_version`: the version of ROCm to use for running the benchmarks.

> - `script_repo`: the repository containing the benchmark scripts.

> - `result_path`: the path where the benchmark results will be stored.

> - `result_repo`: the repository where the benchmark results will be pushed for comparison.

For more details, please refer to the [benchmark.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/benchmark.yaml) file in the repository.

---

## `ci.yaml`

<p>
Overall, this workflow automates the process of building and testing the AMDMIGraphX project across multiple platforms and versions.
</p>

- ## Trigger
The workflow is triggered by the following events:

> - A pull request being opened, synchronized or closed.

> - On push to the `develop`, `master`, and `release/**` branches.

- ## Jobs
The following jobs are executed in the workflow:
> - `cancel`: This job is responsible for canceling any previous runs of the workflow that may still be running. It runs on an `ubuntu-latest` runner and uses the `styfle/cancel-workflow-action` action to cancel any previous runs of the workflow.

> - `tidy`: It runs on an `ubuntu-20.04` runner and runs `clang-tidy` for the codebase in a Docker container with the MIGraphX build environment.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Docker layer cache`: step creates a cache of Docker layers to speed up the build process.
>   - `Restore cache files for tidy`: step restores any previously saved cache files.
>   - `Build the Docker image`: step builds the Docker image with the name `migraphx` using the `hip-clang.docker` Dockerfile.
>   - `Clang tidy`: step runs `clang-tidy` on the source code in the `migraphx` Docker container. This step also compiles the source code and runs tests.
>   - `Clear tidy cache before saving`: step clears the cache before saving it.
>   - `Save cache files for tidy`: step saves the cache files.

> - `cppcheck`: It runs on an `ubuntu-20.04` runner and performs static analysis on code in a Docker container, and caches the results for faster subsequent runs.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Docker layer cache`: step creates a cache of Docker layers to speed up the build process.
>   - `Restore cache files for tidy`: step restores any previously saved cache files.
>   - `Build the Docker image`: step builds the Docker image with the name `migraphx` using the `hip-clang.docker` Dockerfile.
>   - `Cppcheck`: step runs `cppcheck` tool inside a Docker container, passing in the repository code as a volume. It uses CMake to configure the build, and then runs `make` to run the `cppcheck` tool.
>   - `Clear cppcheck  cache before saving`: step clears the cache before saving it.
>   - `Save cache files for cppcheck`: step saves the cache files.

> - `format`: It runs on an `ubuntu-20.04` runner and includes steps for freeing up disk space, caching Docker layers, and checking code formatting.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Docker layer cache`: step creates a cache of Docker layers to speed up the build process.
>   - `Build the Docker image`: step builds the Docker image with the name `migraphx` using the `hip-clang.docker` Dockerfile.
>   - `Check formatting`: step runs a command to check the formatting of the code. It finds all files in the project directory with specific file extensions, excluding those in the `build/` directory.

> - `pyflakes`: It runs on an `ubuntu-20.04` runner and runs the Pyflakes static analysis tool to detect and report Python code issues.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Set up Python`: step sets up the required version of Python for the job.
>   - `Install pyflakes`: step installs the required version of pyflakes and mypy using pip.
>   - `Run pyflakes`: step runs pyflakes and mypy on the specified directories and files.

> - `licensing`: It runs on an `ubuntu-20.04` runner and includes steps to free up space, checkout the code, set up Python and run a license check using a Python script.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Set up Python`: step sets up the required version of Python for the job.
>   - `run License Check`: step runs a Python script named `check_stamped.py` located in the tools directory using the python3 command. The script is responsible for checking the licensing of the project.

---

After already explained jobs we have 2 jobs with multiple matrix configurations, both of them are running on `ubuntu-20.04` runner but right now only `linux` works on all 3 configurations (debug, release, codecov) ,`linux-fpga` works just on (debug).

---
> - `linux`: this job runs continuous integration tests for AMDMIGraphX on a Linux operating system. It tests a variety of build configurations to ensure code quality and compatibility.
>The following steps are executed in this job:
>   - `Free space and install rbuild, lld`: step deletes some directories that may take up space on the runner and installs the LLVM linker (lld) and rbuild.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Set up Python`: step sets up the required version of Python for the job.
>   - `Cache dependencies`: step caches build dependencies using the `actions/cache` action. If the cache already exists for the given dependencies, the job will skip the next step.
>   - `Install dependencies`: step installs dependencies required for the job.
>   - `Restore cache files for ccache`: step restores the ccache cache from a previous build, if it exists.
>   - `Build and test`: step builds and tests the codebase using rbuild. It sets several build flags, including one that enables the use of ccache, and another that enables coverage testing when the `codecov` configuration is used.
>   - `Clear ccache cache before saving`: step is a workaround for a limitation of GitHub Actions caching, and clears the ccache cache before saving it to the cache store.
>   - `Save cache files for ccache`: step saves the ccache cache to the cache store for future use.
>   - `Upload code coverage`: step uploads code coverage results to Codecov when the `codecov` configuration is used. It uses the `lcov` tool to generate coverage reports, and the `codecov` uploader to upload them.

> - `linux-fpga`: this job builds and tests AMDMIGraphX on a Linux operating system with support for FPGA acceleration. It includes additional steps to verify FPGA functionality and performance.
>The following steps are executed in this job:
>   - `Free space`: step deletes some directories that may take up space on the runner.
>   - `actions/checkout@v3`: step checks out the codebase from the repository.
>   - `Set up Python`: step sets up the required version of Python for the job.
>   - `Cache dependencies`: step caches build dependencies using the `actions/cache` action. If the cache already exists for the given dependencies, the job will skip the next step.
>   - `Install dependencies`: step installs build dependencies using rbuild, if the dependencies were not found in the cache.
>   - `Restore cache files for ccache`: step restores the ccache cache from a previous build, if it exists.
>   - `Build and test`: step builds and tests the codebase using rbuild. It sets several build flags, including one that enables the use of ccache, and another that enables coverage testing when the `codecov` configuration is used.
>   - `Clear ccache cache before saving`: step is a workaround for a limitation of GitHub Actions caching, and clears the ccache cache before saving it to the cache store.
>   - `Save cache files for ccache`: step saves the ccache cache to the cache store for future use.
>- Comment out step for now: 
>   - `Upload code coverage`: step uploads code coverage results to Codecov when the `codecov` configuration is used. It uses the `lcov` tool to generate coverage reports, and the `codecov` uploader to upload them.

For more details, please refer to the [ci.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/ci.yaml) file in the repository.

---
## `clean-closed-pr-caches.yaml`

<p>
This workflow has purpose to clean up any cached data related to the pull request.
</p>

- ## Trigger
The workflow is triggered by the following events:

> - A pull request being closed.

- ## Jobs
The workflow has a single job named `cleanup`. The following steps are executed in this job:
> - `Check out code`: step checks out the codebase from the repository.

> - `Cleanup`: step performs the actual cache cleanup using a series of commands.

For more details, please refer to the [clean-closed-pr-caches.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/clean-closed-pr-caches.yaml) file in the repository.

---

## `history.yaml`

<p>
This workflow generates a report of the MiGraphX benchmark results between two dates and sends it to a specified email address. The report is also uploaded to a specified repository.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [history.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/history.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `start_date`: Start date for results analysis.

> - `end_date`: End date for results analysis.

> - `history_repo`: Repository for history results between dates.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `organization`: Organization based on which location of files will be different.

For more details, please refer to the [history.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/history.yaml) file in the repository.

---
## `performance.yaml`

<p>
This workflow runs performance tests on the MIGraphX repository and generates a report of the results.
</p>

- ## Trigger
The workflow will run reusable workflow [perf-test.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/perf-test.yml) by the following events:

> - Pull requests opened, synchronized or closed on the `develop` branch.

> - Schedule: Runs every day of the week from Monday to Saturday at 6:00 AM.

> - Manual trigger through the "Run workflow" button in the Actions tab of the repository.

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm version to use for the performance tests.

> - `performance_reports_repo`: Repository where the performance reports are stored.

> - `benchmark_utils_repo`: Repository where the benchmark utilities are stored.

> - `organization`: Organization based on which location of files will be different.

> - `result_number`: Last N results.

> - `model_timeout`: If a model in the performance test script passes this threshold, it will be skipped.

> - `flags`: Command line arguments to be passed to the performance test script. Default is `-r`.

For more details, please refer to the [performance.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/performance.yaml) file in the repository.

---
## `rocm-image-release.yaml`

<p>
This workflow builds a Docker image for a specified ROCm release version and pushes it to the specified repository. If image already exists nothing will happen, and there is also option to overwrite existing image.
</p>

- ## Trigger
> The workflow is triggered manually through the "Run workflow" button in the Actions tab of the repository and it will run reusable workflow [rocm-release.yml](https://github.com/ROCmSoftwarePlatform/migraphx-benchmark/blob/main/.github/workflows/rocm-release.yml)

- ## Input Parameters
The workflow requires the following inputs:

> - `rocm_release`: ROCm release version to build Docker image for.

> - `benchmark_utils_repo`: Repository where benchmark utils are stored.

> - `base_image`: Base image for ROCm Docker build.

> - `docker_image`: Docker image name for ROCm Docker build.

> - `build_navi`: Build number for the Navi architecture.

> - `organization`: The organization name used to determine the location of files.

> - `overwrite`: Specify whether to overwrite the Docker image if it already exists.

For more details, please refer to the [rocm-image-release.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/rocm-image-release.yaml) file in the repository.

---

## `sync-onnxrt-main.yaml`

<p>
This workflow updates a file with the latest commit hash then creates a pull request using the updated commit hash and adds labels, assignees, reviewers, and a title and body to describe the changes.
</p>

- ## Trigger
The workflow is triggered by the following events:

> - Schedule: Runs every week on Friday at 05:07 PM.

- ## Jobs
The workflow has a single job named `Update and create pull request`. The following steps are executed in this job:
> - `get_date`: step sets an environment variable to the current date in the format 'YYYY-MM-DD'.

> - `extract_sha1`: step fetches the latest SHA1 commit hash of the HEAD branch of the `microsoft/onnxruntime` repository and sets it as an environment variable.

> - `echo_sha1`: step prints the SHA1 commit hash set in step `extract_sha1`.

> - `actions/checkout@v3`: step checks out the codebase from the repository.

> - `update_file`: step updates a file in the repository with the SHA1 commit hash fetched in step `extract_sha1`.

> - `Make changes to pull request`: step uses the `peter-evans/create-pull-request` action to create a pull request.

For more details, please refer to the [sync-onnxrt-main.yaml](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/blob/develop/.github/workflows/sync-onnxrt-main.yaml) file in the repository.

---
