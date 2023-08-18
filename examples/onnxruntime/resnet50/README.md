# Resnet50 inference with MIGraphX and Onnxruntime

## Description

This example demonstrates how to perform an MIGraphX Python API inference through onnxruntime. The model used here is from Torchvision's pretrained resnet50 model

## Content
- [Basic Setup](#Basic-Setup)
- [**Running this Example**](#Running-this-Example)

## Basic Setup
Before running inference we must first install MIGraphX via the Docker method as it also downloads onnxruntime into the dockerfile created. 

Starting from project root:
```
$ cd AMDMIGraphX
$ docker build -t migraphx .
$ docker --device='/dev/kfd' --device='/dev/dri' <your docker settings and mounts> --group-add video -it migraphx
```

The dockerfile will install the latest supported version of ROCm with all the depandacnies needed for MIGraphX

Once the docker file has been installed and you're inside the folder run

```
$ rbuild develop -d deps -B build
$ cd build 
$ make -j$(nproc) package && dpkg -i *.deb
```

to verify migraphx has been installed correclty in the docker run dpkg --list or dpkg -l

```
$ dpkg -l | grep migraphx 
$ ii  migraphx                      2.7.0                             amd64        AMD's graph optimizer
$ ii  migraphx-dev                  2.7.0                             amd64        AMD's graph optimizer
```

## Running this Example

This directory contains everything needed to perform an inference once MIGraphX has been installed in the docker container
Once you've build and installed MIGraphX as a deb package, go to the examples folder, run the pre-req script to build and install
onnxruntime and then install the approrpaite version of pytorch from project root.

```
$ 
$ cd examples/onnxruntime/resnet50
$ ./prereq_steps.sh
$ pip list | grep onnxruntime
$ python resnet50.py
```

example output:

African grey 0.5883207
kite 0.06284781
goldfinch 0.01847724
macaw 0.014789124
fire screen 0.013297303
resnet50, time = 88.89 ms
