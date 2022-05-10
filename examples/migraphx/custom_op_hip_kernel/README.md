# Custom Kernel using MIGraphX API. 
 This is a simple example that shows how to write custom implementation for any operator using MIGraphX's C/C++ APIs. 
 Kernels can be written in either HIP, MIOpen, or by using RocBLAS library. 

 To build the example, ensure ROCm is installed at `/opt/rocm`. 
 1.  `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`
 2.  `cd $MIGRAPHX_SRC/examples/migraphx/custom_op_hip_kernel/`
 3.  `mkdir build && cd build`
 4.  `CXX=/opt/rocm/llvm/bin/clang++ cmake ..  && make`
 5.  `./custom_op_hip_kernel`