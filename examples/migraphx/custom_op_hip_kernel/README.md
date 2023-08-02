# Custom Kernel using MIGraphX API. 
This is an example of a custom operator implementation using MIGraphX's C/C++ APIs. It also demonstrates how to use this custom op in conjunction with rest of MIGraphX operators to build  and run MIGraphX program on GPU. 

Kernels can be written in either HIP, MIOpen, or by using RocBLAS library. This particular example uses **HIP**.

To build the example, ensure ROCm is installed at `/opt/rocm`. 

 1.  Before building this example, find out the gfx-arch of the machine by running `/opt/rocm/bin/rocminfo | grep -o -m 1 "gfx.*"`  pass this gfx-arch as the value of `-DGPU_TARGETS` flag in step 5. Let's assume for now it is MI200 architecture in that case gfx-arch would be `gfx90a`. 
 2.  `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`
 3.  `cd $MIGRAPHX_SRC/examples/migraphx/custom_op_hip_kernel/`
 4.  `mkdir build && cd build`
 5.  `CXX=/opt/rocm/llvm/bin/clang++ cmake .. -DGPU_TARGETS=gfx90a  && make`
 6.  `./custom_op_hip_kernel`