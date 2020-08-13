# AMD MIGraphX

AMD's graph optimization engine.

## Prerequisites
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onxx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) - for python bindings

## Installing the dependencies

Dependencies can be installed using the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).

To install rbuild:
```
pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
```

To build dependencies along with MIGraphX
```
rbuild build -d depend --cxx=/opt/rocm/bin/hcc
```
This builds dependencies in the subdirectory named depend and then builds MIGraphX using these dependencies.

## Building MIGraphX from source

## Configuring with cmake

First create a build directory:


```
mkdir build; 
cd build;
```

Next configure cmake. The hcc compiler is required to build the MIOpen backend:


```
CXX=/opt/rocm/bin/hcc cmake ..
```

If the dependencies from `install_deps.cmake` was installed to another directory, the `CMAKE_PREFIX_PATH` needs to be set to what `--prefix` was set to from `install_deps.cmake`:


```
CXX=/opt/rocm/bin/hcc cmake -DCMAKE_PREFIX_PATH=/some/dir ..
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
