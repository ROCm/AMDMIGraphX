# AMD MIGraphX

AMD MIGraphX is AMD's graph inference engine that accelerates machine learning model inference. AMD MIGraphX can be used by
installing binaries directly or building from source code.

In the following, instructions of how to build and install MIGraphX are described with Ubuntu as the OS
(Instructions of installation on other Linux OSes will come later). Note that all the following instructions assume 
ROCm has been installed successfully. ROCm installation instructions are explained in the [ROCm installation
guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

## Installing from binaries
With ROCm installed correctly, MIGraphX binaries can be installed on Ubuntu with the following command:
```
sudo apt update && sudo apt install -y migraphx
```
then the header files and libs are installed under `/opt/rocm-<version>`, where `<version>` is the ROCm version.

## Building from source

There are three ways to build the MIGraphX sources. 
* [Use the ROCm build tool](#use-the-rocm-build-tool-rbuild)
    
    This approach uses [rbuild](https://github.com/RadeonOpenCompute/rbuild) to install the prerequisites and
build the libs with just one command. 

* [Use cmake](#use-cmake-to-build-migraphx)
    
    This approach uses a script to install the prerequisites, then use cmake to build the source.
      
* [Use docker](#use-docker)
    
    This approach builds a docker image with all prerequisites installed, then build the MIGraphX sources inside a docker container. 

In the following, we will first list the prerequisites required to build MIGraphX source code, then describe 
each of the three approaches.

### List of prerequisites
The following is a list of prerequisites required to build MIGraphX source. 

* [ROCm cmake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onnx](https://github.com/onnx/onnx) files
* [Half](http://half.sourceforge.net/) - IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) - for python bindings
* [JSON](https://github.com/nlohmann/json) - for model serialization to json string format
* [MessagePack](https://msgpack.org/index.html) - for model serialization to binary format
* [SQLite3](https://www.sqlite.org/index.html) - to create database of kernels' tuning information or execute queries on existing database

#### Use the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).

In this approach, we use the [rbuild](https://github.com/RadeonOpenCompute/rbuild) build tool to
build MIGraphX. The specific steps are as follows:

1) Install rocm-cmake, pip3, rocblas, and miopen-hip with the command

```
sudo apt install -y rocm-cmake python3-pip rocblas miopen-hip
```

2) Install [rbuild](https://github.com/RadeonOpenCompute/rbuild) (sudo may be required here.)

```
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
```

3) Build MIGraphX source code

```
rbuild build -d depend -B build
```

then all the prerequisites are in the folder `depend`, and MIGraphX is built in the `build` directory.

Also note that you may meet the error of `rbuild: command not found`. It is because rbuild is installed 
at `$HOME/.local/bin`, which is not in `PATH`. You can either export PATH as `export PATH=$HOME/.local/bin:$PATH` 
to add the folder to `PATH` or add the option `--prefix /usr/local` in the pip3 command when installing rbuild.

#### Use cmake to build MIGraphX

If using this approach, we need to install the prerequisites, configure the cmake, and then build the source.

##### Installing the prerequisites

For convenience, the prerequisites can be built automatically with rbuild as:

```
rbuild prepare -d depend
```

then all the prerequisites are in the folder `depend`, and they can be used in the `cmake` configuration
as `-DCMAKE_PREFIX_PATH=depend`.

If you have sudo access, as an alternative to the rbuild command, you can install the prerequisites just 
like in the dockerfile by calling `./tools/install_prereqs.sh`.

(Note that this script is for Ubuntu. By default, all prerequisites are installed at the default location `/usr/local` 
and are accessible by all users. For the default location, `sudo` is required to run the script.
You can also specify a location at which the prerequisites are installed with `./tools/install_prereqs.sh $your_loc`.)

##### Building MIGraphX source and install libs

With the above prerequisites installed, we can build source as:

1) Go to the project folder and create a `build` directory:


```
mkdir build
cd build
```

2) Configure the cmake. If the prerequisites are installed at the default location `/usr/local`, the command is:

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

#### Use docker

The easiest way to setup the development environment is to use docker. With the dockerfile, you can build a docker image as:

    docker build -t migraphx .

Then to enter the developement environment use `docker run`:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/code/AMDMIGraphX -w /code/AMDMIGraphX --group-add video -it migraphx

In the docker container, all the required prerequisites are already installed, so users can just go to the folder 
`/code/AMDMIGraphX` and follow the steps in the above [Build MIGraphX source and install
libs](#building-migraphx-source-and-install-libs)
section to build MIGraphX source.

### Using MIGraphX Python Module
To use MIGraphX's Python module, please either set `PYTHONPATH` or use `.deb` package as explained below:

- Setting `PYTHONPATH` :
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```
- Creating and installing the package:

To create deb package:
```
make package
```
This will provide the path of .deb package.

To install:
```
dpkg -i <path_to_deb_file>
```

### Calling MIGraphX APIs
To use MIGraphX's C/C++ API in your cmake project, we need to set `CMAKE_PREFIX_PATH` to the MIGraphX
installation location and then do 
```
find_package(migraphx)
target_link_libraries(myApp migraphx::c)
```
Where `myApp` is the cmake target in your project.

## Building for development

Using rbuild, the dependencies for development can be installed with:

```
rbuild develop
```

This will install the dependencies for development into the `deps` directory and
configure `cmake` to use those dependencies in the `build` directory. These
directories can be changed by passing the `--deps-dir` and `--build-dir` flags
to `rbuild` command:

```
rbuild develop --build-dir build_rocm_55 --deps-dir /home/user/deps_dir
```

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the docs/html folder.

Documentation is built using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core)

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Depending on your setup `sudo` may be required for the pip install.

## Formatting the code

All the code is formatted using clang-format. To format a file, use:

```
clang-format-10 -style=file -i <path-to-source-file>
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```
