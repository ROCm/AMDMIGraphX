# AMD MIGraphX

AMD's graph optimization engine.

## Prerequisites
* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onxx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library

## Installing the dependencies

The dependencies can be installed with the `install_deps.cmake`, script: `cmake -P install_deps.cmake`.

This will install by default to `/usr/local` but it can be installed in another location with `--prefix` argument:

```
cmake -P install_deps.cmake --prefix /some/local/dir
```


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
