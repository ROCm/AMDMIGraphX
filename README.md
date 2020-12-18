# AMD MIGraphX

AMD's graph optimization engine.

## Prerequisites
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onnx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) - for python bindings
* [ONNX](https://github.com/onnx/onnx) and [Pytest](https://github.com/pytest-dev/pytest) for running the ONN backend test


## Installing the dependencies
There are two alternative ways to install the dependencies required by MIGraphX:

* Dependencies can be installed one by one using a [script](https://github.com/mvermeulen/rocm-migraphx/blob/master/scripts/build_prereqs.sh).
(Note: 1. You may need the sudo to install the above dependencies. 2. rocBLAS and MIOpen can be installed with the
command ```sudo apt install -y rocblas miopen-hip```)

* Dependencies can also be installed using the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).
They are listed in files requirements.txt and dev-requirements.txt in the project diretory.

To install rbuild (sudo may be needed.):
```
pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
```

To build dependencies along with MIGraphX
* In ROCm3.3:

```
rbuild develop -d depend -B build --cxx=/opt/rocm/bin/hcc
```

* In ROCm3.7 and later releases:

```
rbuild develop -d depend -B build --cxx=/opt/rocm/llvm/clang++
```

This builds dependencies in the subdirectory named "depend" (full path is represented as "$(amdmigraphx_dir)/depend."

Note that if rocBLAS and MIOpen are installed with the command ```sudo apt install -y rocblas miopen-hip```, we can
comment out the two lines "ROCmSoftwarePlatform/rocBLAS@abd98a2b48b29326ebaef471630786a548622c06" and
"ROCmSoftwarePlatform/MIOpen@2.4.0" of the file requirements.txt (adding a '#' character at the start of each line)

## Building MIGraphX from source

First create a build directory:


```
mkdir build; 
cd build;
```

Next configure cmake. The hcc or clang compiler is required to build the MIOpen/HIP backend kernels:

If the [script](https://github.com/mvermeulen/rocm-migraphx/blob/master/scripts/build_prereqs.sh) is called to install
the script, MIGraphX can be build as:
* In ROCm3.3:

```
CXX=/opt/rocm/bin/hcc cmake ..
```
* In ROCm3.7 or later releases:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
```

If the dependencies were installed in the folder "depend" using rbuild, the `CMAKE_PREFIX_PATH` needs to be set to 
the same dependency directory (full path are needed here.) as:
* ROCM3.3:

```
CXX=/opt/rocm/bin/hcc cmake -DCMAKE_PREFIX_PATH=$(amdmigraphx_dir)/depend ..
```
* ROCM3.7 or later releases:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DCMAKE_PREFIX_PATH=$(amdmigraphx_dir)/depend ..
```

Then MIGraphX can be build as:
```
make -j$(nproc)
```
and correctness can be verified as:
```
make -j$(nproc) check
```

#### Changing the cmake configuration

The configuration can be changed after running cmake by using `ccmake`:

` ccmake .. ` **OR** `cmake-gui`: ` cmake-gui ..`

## Building the library

The library can be built, from the `build` directory using the 'Release' configuration:

` cmake --build . --config Release ` **OR** ` make `

And can be installed by using the 'install' target:

` cmake --build . --config Release --target install ` **OR** ` make install `

This will install the library to the `CMAKE_INSTALL_PREFIX` path that was set.

To build a debug version of the library, the cmake variable `CMAKE_BUILD_TYPE` can be set to `Debug`.

` cmake -DCMAKE_BUILD_TYPE=Debug . `

## Running the tests

The tests can be run by using the 'check' target:

` cmake --build . --config Release --target check ` **OR** ` make check `

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the doc/html folder.

Documentation is built using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html), [Sphinx](http://www.sphinx-doc.org/en/stable/index.html), and [Breathe](https://breathe.readthedocs.io/en/latest/)

Requirements for both Sphinx and Breathe can be installed with:

`pip install -r doc/requirements.txt`

Depending on your setup `sudo` may be required for the pip install.

## Formatting the code

All the code is formatted using clang-format. To format a file, use:

```
clang-format-5.0 -style=file -i <path-to-source-file>
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Using docker

The easiest way to setup the development environment is to use docker. You can build the top-level docker file:

    docker build -t migraphx .

Then to enter the developement environment use `docker run`:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/data -w /data --group-add video -it migraphx
