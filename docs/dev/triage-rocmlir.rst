.. meta::
  :description: Issue Triaging Guide for suspected issues
  :keywords: MIGraphX, rocMLIR, issues, pipeline, compilation, bug, code base, kernel, contribution, developing

Issue Triaging Guide for suspected rocMLIR issue
================================================

This document serves as a guide to narrow down whether a bug is due to rocMLIR backend of MIGraphX.

There are broadly 3 categories of bugs that can be due to rocMLIR.

1. ``[B1]`` rocMLIR compilation bug
2. ``[B2]`` rocMLIR runtime failure
3. ``[B3]`` accuracy issue from rocMLIR

Step 1 - Use ``MIGRAPHX_DISABLE_MLIR=1``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First see if the bug persists after disabling MLIR. If so, it is highly likely
it is not a MLIR bug but rather a MIGraphX bug.

If you dont see a failure, please proceed.

Step 2 - See if its a ``B1`` - rocMLIR compilation bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLIR has two pipelines that are used to generate a kernel: highlevel 
pipeline and backend pipeline

Step 2.1 
~~~~~~~~
If the highlevel pipeline fails, you should see error that starts with

``Invalid MLIR created:``

If error message like above is present then proceed with following steps. 

1. Please disable threading first by setting environment variable MIGRAPHX_GPU_COMPILE_PARALLEL=1. 
2. Set environment variable ``MIGRAPHX_TRACE_MLIR=1`` . 
3. Create a temporary directory to store MXR files. Set environment variable ``MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/``
4. Run the model and pipe the logs to a file. You should see MLIR module printed just before the failure. 
   For example, it would look like as follows.

::

   module {
     func.func @mlir_convolution(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
       %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
       return %0 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
     }
   }

5. Provide MLIR module printed just before the failure from previous step to rocMLIR team to debug. Please provide only the **SINGLE** module that fails and not the entire log for rocMLIR team figure out which is failing.
6. Set ``MIGRAPHX_TRACE_MLIR=1`` . Run each individual MXR file from MXR dump directory with ``migraphx-driver``. Find out which MXR file is failing to compile. Failing MXR file must have exacty **the same MLIR** module as the one from the original model identified in previous step. Provide this MXR file to rocMLIR team. 

Step 2.2 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the backend pipeline fails for *all solutions*, you should see error that starts with

``No solutions provided for mlir_*`` or
``No valid tuned compilation for mlir_*``

NOTE 1 : Currently MIGraphX does not run applicability pipeline of
rocMLIR. Therefore, you will see them for inapplicable tuning params as
well.

NOTE 2 : Ignore ``MLIR backend compilation failed: "`` if you don't see
``No solutions provided for mlir_*``

If the above error message is present then proceed with following steps.

1. Please disable threading first using ``MIGRAPHX_GPU_COMPILE_PARALLEL=1``. 
2. Set environment variable ``MIGRAPHX_TRACE_MLIR=2``
3. Create a temporary directory to store MXR files. Set environment variable ``MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/``
4. Run the model and pipe the logs to a file. You should see MLIR module printed just before the failure. For example, it would look like as follows. 

::

   module {
     func.func @mlir_convolution(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel, arch = ""} {
       %0 = memref.alloc() : memref<1x1x30x30x16xf32>
       rock.conv(%arg1, %arg0, %0) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["gi", "ni", "0i", "1i", "ci"], output_layout = ["go", "no", "0o", "1o", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
       %4 = memref.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [1, 1, 1, 1, 16] : memref<16xf32> into memref<1x1x1x1x16xf32>
       linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %4 : memref<1x1x30x30x16xf32>, memref<1x1x1x1x16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
       ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
         %8 = arith.addf %arg4, %arg5 : f32
         linalg.yield %8 : f32
       }
       return
     }
   }

5. Provide MLIR module printed just before the failure from previous step to rocMLIR team to debug. Please provide only the SINGLE module that fails and not the entire log for rocMLIR team figure out which is failing.
6. Set ``MIGRAPHX_TRACE_MLIR=2`` and ``MIGRAPHX_GPU_COMPILE_PARALLEL=1`` . Run each individual MXR file from MXR dump directory with migraphx-driver. Find out which MXR file is failing to compile. Failing MXR file must have exacty the same MLIR module as the one from the original model identified in previous step. Provide this MXR file to rocMLIR team. 

Step 3 See if its a ``B2`` - runtime failure of MLIR-generated kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each individual kernels are run in the benchmarking process of MIGraphX. Therefore, set ``MIGRAPHX_TRACE_BENCHMARKING=3``

You should see messages like following when compiling model. 

::

   Benchmarking mlir_* : xx configs
   Problem: <problem key>
   Benchmarking solution: <perf config>

In this case there are two things to indentify. (1) MLIR module that is causing the runtime failure (2) PerfConfig which that used when compiling MLIR module.

Follow these steps indentify those two things. 

1. Set ``MIGRAPHX_TRACE_BENCHMARKING=3``
2. Set ``MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created``
3. Run model. You should see MLIR module just before MIGraphX starts benchmarking.  Please provide that to rocMLIR team. 
4. Just before the failure you should also see ``Benchmarking solution : <perf_config>``. Take note of that and pass that to rocMLIR team.
5. Set ``MIGRAPHX_TRACE_BENCHMARKING=3`` and Run each individual MXR file from temporary dump directory to see which one is failing. 
    a. For the failing MXR, just before the start of the benchmarking process it should print MLIR module. It must be the same as the one identified earlier in step (3). 
    b. Before the failure it would print ``Benchmarking Solution: <perf_config>``. It must be the same as the one identified from step (4). 
    c. Provide the failing MXR file to rocMLIR team to investigate. 
6. Note down whether model was compiled using ``--exhaustive-tune`` or not. Mention that in ticket to rocMLIR.


Step 4 See if its a ``B3`` - accuracy issue of MLIR-generated kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Create a temporary directory to store MXR files. Set environment variable ``MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/``
2. Set ``MIGRAPHX_TRACE_MLIR=1`` and ``MIGRAPHX_GPU_COMPILE_PARALLEL=1``. Run each individual MXR file from MXR dump directory with ``migraphx-driver verify`` to find out which one is failing the accuracy. 

3. Provide MLIR module from failing MXR to rocMLIR team. For example, it would look something like following

::

 module {
     func.func @mlir_dot_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
       %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
       %1 = migraphx.add %0, %arg2 : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
       return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
     }
   }

4. Provide failing MXR file to rocMLIR team. 