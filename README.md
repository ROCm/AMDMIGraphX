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
then the head files and libs are located at `/opt/rocm/include` and `/opt/rocm/lib`, respectively, which can be
included and linked by adding the corresponding folders to the Makefile.

## Building from source

There are two ways to build the MIGraphX sources. One is installing the dependencies, then using 
cmake to build the source. The other is using the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).
In the following, we will first list the dependencies required to build MIGraphX source code, then describe each of the
two approaches.

### List of dependencies
The following is a list of dependencies required to build MIGraphX source. This list is also available in the
requirement files `dev-requirements.txt` and `requirements.txt`.

* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onnx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) - for python bindings

#### Use cmake to build MIGraphX

In this approach, we need to install the dependencies, configure the cmake, and then build the source.

##### Installing the dependencies

You can manually download and installing the above dependencies one by one. For convience, we provide a shell 
script [install_prereqs.sh](./tools/install_prereqs.sh) that can automatically install all the above dependencies with
the command 

```./tools/install_prereqs.sh```

(Note: By default, all dependencies are installed at the default location `/usr/local` 
and are accessible by all users. For the default location, `sudo` is required to run the script.
You can also specify a location at which the dependencies are installed with `./tools/install_prereqs.sh $your_loc`.)

##### Building MIGraphX source and install libs

With the above dependencies installed, we can build source as:

1) First create a build directory:


```
mkdir build
cd build
```

2) Configure the cmake. If the dependencies are installed at the default location `/usr/local`, the command is:

```
CXX=/opt/rocm/llvm/bin/clang++ cmake ..
```
Otherwise, you need to set `-DCMAKE_PREFIX_PATH=$your_loc` to configure the cmake. 

3) Build MIGraphX source code

```
make -j$(nproc)
```

Correctness can be verified as:

```
make -j$(nproc) check
```

MIGraphX libs can be installed as:

```
make install
```

#### Use the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).

In this approach, we need to install the [rbuild](https://github.com/RadeonOpenCompute/rbuild) first, then use it to
build MIGraphX. rbuild can be installed as (sudo may be needed.):
```
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
```
and pip3 can be installed as `sudo apt update && sudo apt install -y python3-pip`.
We also need to install [rocm-cmake](https://github.com/RadeonOpenCompute/rocm-cmake) as `sudo apt install -y rocm-cmake`.

Then MIGraphX can be built as:

```
rbuild build -d depend -B build --cxx=/opt/rocm/llvm/clang++
```

Note that for ROCm3.7 and later release, Ubuntu 18.04 or later releases are needed. Upgrapde to Ubuntu 18.04 can be
done as:

```
sudo apt update
sudo apt install linux-headers-4.18.0-25-generic linux-image-4.18.0-25-generic  linux-modules-4.18.0-25-generic linux-modules-extra-4.18.0-25-generic -y
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

The easiest way to setup the development environment is to use docker. With the docker file, you can build an docker image as:

    docker build -t migraphx .

Then to enter the developement environment use `docker run`:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/code/AMDMIGraphX -w /code/AMDMIGraphX --group-add video -it migraphx

In the docker container, all the required dependencies are already installed, so users can just go to the folder 
`/code/AMDMIGraphX` and follow the steps in the above [Build MIGraphX source and install
libs](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/tree/refine_readme#building-migraphx-source-and-install-libs)
section to build MIGraphX source.

