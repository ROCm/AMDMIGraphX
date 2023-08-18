# Resnet50 inference with MIGraphX and Onnxruntime

## Description

This example demonstrates how to perform an MIGraphX Python API inference through onnxruntime. The model used here is from Torchvision's pretrained resnet50 model

## Content
- [Basic Setup](#Basic-Setup)
- [**Running this Example**](#Running-this-Example)

## Basic Setup
Before running inference we must first install MIGraphX via the Docker method as it also downloads onnxruntime into the dockerfile created.

## Running this Example

This directory contains everything needed to perform an inference once MIGraphX has been installed in the docker contaienr

```
$ mkdir build
$ cd build
$ CXX=/opt/rocm/llvm/bin/clang++ cmake ..
$ make -j$(nproc) package
$ dpkg -i *.deb
```
Once you've build and installed MIGraphX as a deb package go to the examples folder, run the pre-req script to build and install
onnxruntime and then install the approrpaite version of pytorch

```
$ cd examples/onnxruntime/resnet50
$ ./prereq_steps.sh
$ pip list | grep onnxruntime
$ python resnet50.py
```
