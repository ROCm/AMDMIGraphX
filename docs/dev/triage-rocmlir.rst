.. meta::
   :description: Issue Triaging Guide for suspected issues
   :keywords: MIGraphX, rocMLIR, issues, pipeline, compilation, bug, code base, kernel, contribution, developing

*****************************************************
Issue Triaging Guide for Suspected rocMLIR Issues
*****************************************************

This document serves as a guide to narrow down whether a bug is due to the rocMLIR backend of MIGraphX.

Overview
========

There are broadly three categories of bugs that can be due to rocMLIR:

1. **[B1]** rocMLIR compilation bug
2. **[B2]** rocMLIR runtime failure
3. **[B3]** Accuracy issue from rocMLIR

Step 1: Use ``MIGRAPHX_DISABLE_MLIR=1``
=========================================

First, see if the bug persists after disabling MLIR. If so, it is highly likely not a MLIR bug but rather a MIGraphX bug.

If you don't see a failure, please proceed to Step 2.

Step 2: Identify ``[B1]`` - rocMLIR Compilation Bug
====================================================

MLIR has two pipelines that are used to generate a kernel:

- **Highlevel pipeline**
- **Backend pipeline**

Step 2.1: Highlevel Pipeline Failure
--------------------------------------

If the highlevel pipeline fails, you should see an error that starts with:

.. code-block:: text

   Invalid MLIR created:

If an error message like the above is present, then proceed with the following steps:

1. Disable threading first by setting the environment variable:

   .. code-block:: bash

      export MIGRAPHX_GPU_COMPILE_PARALLEL=1

2. Set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_TRACE_MLIR=1

3. Create a temporary directory to store MXR files and set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/

4. Run the model and pipe the logs to a file. You should see the MLIR module printed just before the failure. For example:

   .. code-block:: mlir

      module {
        func.func @mlir_convolution(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
          %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
          return %0 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
        }
      }

5. Provide the MLIR module printed just before the failure from the previous step to the rocMLIR team to debug. 

   .. important::
      Please provide only the **SINGLE** module that fails and not the entire log for the rocMLIR team to figure out which is failing.

6. Set ``MIGRAPHX_TRACE_MLIR=1`` and run each individual MXR file from the MXR dump directory with ``migraphx-driver``. Find out which MXR file is failing to compile. The failing MXR file must have exactly **the same MLIR** module as the one from the original model identified in the previous step. Provide this MXR file to the rocMLIR team.

Step 2.2: Backend Pipeline Failure
------------------------------------

If the backend pipeline fails for **all solutions**, you should see an error that starts with:

.. code-block:: text

   No solutions provided for mlir_*

or

.. code-block:: text

   No valid tuned compilation for mlir_*

.. note::

   Currently MIGraphX does not run the applicability pipeline of rocMLIR. Therefore, you will see these errors for inapplicable tuning params as well.
   Also, you can ignore ``MLIR backend compilation failed:`` as long as you don't see ``No solutions provided for mlir_*``

If the above error message is present, then proceed with the following steps:

1. Disable threading first by setting:

   .. code-block:: bash

      export MIGRAPHX_GPU_COMPILE_PARALLEL=1

2. Set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_TRACE_MLIR=2

3. Create a temporary directory to store MXR files and set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/

4. Run the model and pipe the logs to a file. You should see the MLIR module printed just before the failure. For example:

   .. code-block:: mlir

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

5. Provide the MLIR module printed just before the failure from the previous step to the rocMLIR team to debug. 

   .. important::
      Please provide only the **SINGLE** module that fails and not the entire log for the rocMLIR team to figure out which is failing.

6. Set ``MIGRAPHX_TRACE_MLIR=2`` and ``MIGRAPHX_GPU_COMPILE_PARALLEL=1``. Run each individual MXR file from the MXR dump directory with ``migraphx-driver``. Find out which MXR file is failing to compile. The failing MXR file must have exactly the same MLIR module as the one from the original model identified in the previous step. Provide this MXR file to the rocMLIR team.

Step 3: Identify ``[B2]`` - Runtime Failure of MLIR-Generated Kernel
======================================================================

Each individual kernel is run in the benchmarking process of MIGraphX. Therefore, set:

.. code-block:: bash

   export MIGRAPHX_TRACE_BENCHMARKING=3

You should see messages like the following when compiling the model:

.. code-block:: text

   Benchmarking mlir_* : xx configs
   Problem: <problem key>
   Benchmarking solution: <perf config>

In this case, there are two things to identify:

1. MLIR module that is causing the runtime failure
2. PerfConfig which was used when compiling the MLIR module

Follow these steps to identify those two things:

1. Set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_TRACE_BENCHMARKING=3

2. Set the MXR dump directory:

   .. code-block:: bash

      export MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/

3. Run the model. You should see the MLIR module just before MIGraphX starts benchmarking. Please provide that to the rocMLIR team.

4. Just before the failure, you should also see ``Benchmarking solution : <perf_config>``. Take note of that and pass it to the rocMLIR team.

5. Set ``MIGRAPHX_TRACE_BENCHMARKING=3`` and run each individual MXR file from the temporary dump directory to see which one is failing:

   a. For the failing MXR, just before the start of the benchmarking process, it should print the MLIR module. It must be the same as the one identified earlier in step 3.
   
   b. Before the failure, it would print ``Benchmarking Solution: <perf_config>``. It must be the same as the one identified from step 4.
   
   c. Provide the failing MXR file to the rocMLIR team to investigate.

6. Note down whether the model was compiled using ``--exhaustive-tune`` or not. Mention that in the ticket to rocMLIR.

Step 4: Identify ``[B3]`` - Accuracy Issue of MLIR-Generated Kernel
=====================================================================

1. Create a temporary directory to store MXR files and set the environment variable:

   .. code-block:: bash

      export MIGRAPHX_MLIR_DUMP_TO_MXR=/path/to/temp/dir/created/

2. Set the following environment variables and run each individual MXR file from the MXR dump directory with ``migraphx-driver verify`` to find out which one is failing the accuracy:

   .. code-block:: bash

      export MIGRAPHX_TRACE_MLIR=1
      export MIGRAPHX_GPU_COMPILE_PARALLEL=1

3. Provide the MLIR module from the failing MXR to the rocMLIR team. For example:

   .. code-block:: mlir

      module {
        func.func @mlir_dot_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes {arch = "gfx90a:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 110 : i64} {
          %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
          %1 = migraphx.add %0, %arg2 : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
          return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
        }
      }

4. Provide the failing MXR file to the rocMLIR team.
