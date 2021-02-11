# AMD MIGraphX

AMD MIGraphX is AMD's graph inference engine to accelerate model inference on AMD GPUs. AMD MIGraphX can be used by
installing binaries directly or building from source code.

Note that all the following instructions are based on that ROCm has been installed successfully. ROCm installation
instructions are explained in the [ROCm installation
guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

## Installing from binaries
With ROCm installed correctly, MIGraphX binaries can be installed with the following command:
```
sudo apt update && sudo apt install -y migraphx
```
then the head files and libs are located at ```/opt/rocm/include``` and ```/opt/rocm/lib```, respectively, which can be
included and linked by adding the corresponding folders to the Makefile.

## Building from source

Building MIGraphX from sources needs dependencies, which must be installed before building the source code. In the
following, we first list the dependencies, and then explain two ways of installing them.

### List of dependencies
The following is a list of dependencies required to build MIGraphX from source code. This list is also available in the
requirement files ```dev-requirements.txt``` and ```requirements.txt```.

* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onnx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) - for python bindings
* [ONNX 1.7.0](https://github.com/onnx/onnx) and [Pytest](https://github.com/pytest-dev/pytest) for running the ONN backend
  tests 

Note: 1) We have to use ONNX version 1.7.0 since changes in ONNX 1.8.0 is incompatible with our current implementation. 
ONNX 1.7.0 can be installed as ```pip3 install onnx==1.7.0```, and updates to support ONNX version 1.8.0 will come soon. 2)
MIOpen and rocBLAS can be installed as ```sudo apt update && sudo apt install -y miopen-hip rocblas```.

### Installing the dependencies
There are two alternative ways to install the above dependencies:

#### Run a shell script [build_prereqs.sh](./tools/build_prereqs.sh).

(Note: 1. You need the sudo to install the above dependencies. 2. All dependencies are installed in default locations in
the system and are accessible by all users.)

#### Use the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).

To install rbuild (sudo may be needed.):
```
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
```
and pip3 can be installed as `sudo apt update && sudo apt install -y python3-pip`

Dependencies are listed in files `requirements.txt` and `dev-requirements.txt` in the project diretory. To build the
dependencies,
* In ROCm3.3:

```
rbuild develop -d depend -B build --cxx=/opt/rocm/bin/hcc
```

* In ROCm3.7 and later releases:

```
rbuild develop -d depend -B build --cxx=/opt/rocm/llvm/clang++
```

This builds dependencies in the folder "depend".

Note that if rocBLAS and MIOpen are installed with the command ```sudo apt install -y rocblas miopen-hip``` as mentioned
above, we can comment out the two lines 
```
ROCmSoftwarePlatform/rocBLAS@abd98a2b48b29326ebaef471630786a548622c06
ROCmSoftwarePlatform/MIOpen@2.4.0
```
in the file `requirements.txt` (adding a '#' character at the start of each line).
Also note that for ROCm3.7 and later release, Ubuntu 18.04 or later releases are needed. Upgrapde to Ubuntu 18.04 can be
done as:

```
sudo apt update
sudo apt install linux-headers-4.18.0-25-generic linux-image-4.18.0-25-generic  linux-modules-4.18.0-25-generic linux-modules-extra-4.18.0-25-generic -y
```

### Building MIGraphX source

First create a build directory:


```
mkdir build; 
cd build;
```

Next, configure the cmake. If the script [buiild_prereqs.sh](./tools/build_prereqs.sh) is used to install the
dependencies, then all dependencies are installed at default locations, and MIGraphX can be build as:
* In ROCm3.3:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
```

* In ROCm3.7 or later releases:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
```

If the above rbuild command was used to build and install the dependencies, then all dependencies are in the folder
"depend", and the `CMAKE_PREFIX_PATH` needs to be set to the same folder "depend" (full path are needed here.), and the
command is:

* ROCM3.3:

```
CXX=/opt/rocm/bin/hcc cmake -DCMAKE_PREFIX_PATH=../depend ..

```
* ROCM3.7 or later releases:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake -DCMAKE_PREFIX_PATH=../depend ..
```

Then we can build MIGraphX source code as:

```
make -j$(nproc)
```

and correctness can be verified as:

```
make -j$(nproc) check
```

MIGraphX libs can be installed as:

```
make install
```

### Building the documentation

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

The easiest way to setup the development environment is to use docker. Docker files for different versions of ROCm are
provided. ```hcc.docker``` is for ROCm3.3, and ```hip-clang.docker``` is for ROCm3.7.

With the docker files, you can build an docker image as:

    docker build -t migraphx -f hcc.docker (or hip-clang.docker) .

Then to enter the developement environment use `docker run`:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/data -w /data --group-add video -it migraphx

In the docker container, all the required dependencies are installed, then users can either install MIGraphX
binaries or build from the source code following the steps with the dependencies installed using the script
[buiild_prereqs.sh](./tools/build_prereqs.sh). In the docker container, all dependencies are installed at 
default location, and there is no need to set the ```CMAKE_PREFIX_PATH``` when configuring the cmake.

