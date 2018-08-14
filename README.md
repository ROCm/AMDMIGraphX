# MIGraph

AMD's library for graph optimizations.

## Prerequisites
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onxx](https://github.com/onnx/onnx) files

## Installing the dependencies

The dependencies can be installed with the `install_deps.cmake`, script: `cmake -P install_deps.cmake`.

This will install by default to `/usr/local` but it can be installed in another location with `--prefix` argument:

```
cmake -P install_deps.cmake --prefix /some/local/dir
```


## Building MIGraph from source

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

## Running the tests

The tests can be run by using the 'check' target:

` cmake --build . --config Release --target check ` **OR** ` make check `

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

The generated documentation will be located in `doc/doxygen/`.

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

    docker build -t migraph .

Then to enter the developement environment use `docker run`:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/data -w /data --group-add video -it migraph
