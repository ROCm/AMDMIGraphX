.. meta::
  :description: Recompile the GPU kernels of a compiled MIGraphX program for the same or another GPU
  :keywords: MIGraphX, recompile, recompilable, retarget, mxr, cross-compile, exhaustive tune, gfx, ROCm

.. _recompile-mxr:

********************************************************************
Recompile an MXR file for another GPU
********************************************************************

An ``.mxr`` file is a compiled MIGraphX program. Its GPU kernels are built for one GPU
architecture, for example ``gfx90a``. To run the model on another GPU, or to tune it again on the same
GPU, you normally compile it again from the original model, which you might not have.

**Recompile** rebuilds the kernels of an ``.mxr`` file for the same GPU or for another one, starting from the
``.mxr`` file itself. No ONNX model is needed. Only the kernels change. The graph, the weights (literals), the
memory plan, and the stream schedule stay as they were compiled.

You can use recompile to:

* **Retune on the same GPU**, for example with exhaustive tuning, without compiling the model again.
* **Move an MXR file to another GPU** of a newer or an older generation, for example ``gfx90a`` to ``gfx942``,
  ``gfx1100`` to ``gfx1201``, or ``gfx1151`` to ``gfx942``.
* **Cross-compile** for a GPU that is not in the machine, with ``--gpu-arch``. See :ref:`cross-compilation`.

To make this possible, compile the model once with ``--recompilable``. That keeps a small *recipe* for each
kernel inside the ``.mxr`` file and records which GPU the file was compiled for.

.. code-block:: text

    model.onnx
        |  migraphx-driver compile --gpu --recompilable
        v
    model_A.mxr          kernels and recipes for GPU A
        |  migraphx-driver recompile [--gpu-arch B]
        v
    model_B.mxr          kernels for GPU B
        |  migraphx-driver perf | run | verify   (on GPU B)
        v

.. note::

   Recompile is available from ``migraphx-driver`` and the C++ library API only. There is no C API, Python API, or
   public C++ wrapper API for it. It supports the ``gpu`` target only.

.. _recompile-prerequisites:

Prerequisites
====================================================================

* An ``.mxr`` file compiled with ``--recompilable`` by the same MIGraphX build that runs ``recompile``. Files
  compiled without it, or by an older build, are refused and must be compiled again from the model.

``--recompilable`` also changes how the model is compiled, so that as many kernels as possible are ones that
recompile can rebuild. It does this on its own. You do not need to set any environment variables.

* Dot, convolution, and fused versions of them are compiled with rocMLIR instead of rocBLAS, hipBLASLt, or MIOpen,
  because rocMLIR kernels have recipes. This can change the performance of these operators.
* LRN is rewritten into MIGraphX operators instead of calling the MIOpen LRN, so it also gets a recipe.

.. note::

   rocMLIR must be enabled in the build. Without it, dot and convolution stay library calls with no recipe, and
   they can be recompiled on the same GPU only. See :ref:`recompile-limitations`.

Recompile a model
====================================================================

Step 1: Compile the model with ``--recompilable``
--------------------------------------------------------------------

.. code-block:: shell

   migraphx-driver compile model.onnx --gpu --recompilable -o model.mxr

This compiles for the GPU in the machine. To compile for a different GPU, add ``--gpu-arch <arch>``, for example
``--gpu-arch gfx90a``. Pass the same input options that you normally use, for example ``--input-dim`` or ``--fp16``.

Step 2: Check what the file contains
--------------------------------------------------------------------

.. code-block:: shell

   migraphx-driver read model.mxr

``read`` opens the file without loading its kernels, so it works on a machine that does not have the GPU that
the file was compiled for. Check two things:

* The main module starts with ``@comment[gpu_compile_options]`` and ``@comment[gpu_target]``.
* The ``gpu::code_object`` kernels end with ``[main:recipeN]``.

If either is missing, the file cannot be recompiled. See :ref:`recompile-read-output` for how to read the output.

Step 3: Recompile
--------------------------------------------------------------------

For the GPU in the machine. If the file was compiled for another GPU, this moves it to the GPU in the machine. If it
was compiled for this GPU, this tunes it again:

.. code-block:: shell

   migraphx-driver recompile model.mxr --gpu -o model_this_gpu.mxr

Or cross-compile for a GPU that is not in the machine:

.. code-block:: shell

   migraphx-driver recompile model.mxr --gpu --gpu-arch gfx1201 -o model_gfx1201.mxr

The ``-o`` option is required. If the recompile fails, nothing is written and the input file is not changed.

The command prints a short report:

.. code-block:: text

   [INFO] Recompiling ...
   [INFO] recompile: recompiling 57 kernels for gfx1151
   [INFO] recompile: left as compiled: gpu::hip_gemm
   [INFO] note: graph decisions were made for gfx90a; compile the model again for best performance on gfx1151.

* ``recompiling N kernels``: the number of kernels that have a recipe and are rebuilt.
* ``left as compiled``: instructions that have no recipe, so recompile cannot rebuild them. These are library calls
  such as rocBLAS, hipBLASLt or MIOpen, CK kernels, and any kernel that has no recipe link. This line only appears
  for a recompile on the same GPU. They were built for this GPU, so they are still valid, and recompile keeps them
  unchanged while it rebuilds the other kernels. On a different GPU they cannot be reused, so recompile stops
  instead. See :ref:`recompile-limitations` for when a file can have them.
* ``note: graph decisions ...``: printed when the GPU generation changes. See :ref:`recompile-limitations`.

These options change how the recompile runs:

.. list-table::
   :header-rows: 1

   * - Option
     - Effect
   * - ``--exhaustive-tune``
     - Tries every kernel candidate instead of the quick set. The options that the file was compiled with are not
       reused, so pass this option again on each recompile where you want it.
   * - ``--backend-options``
     - Target-specific options as a JSON object. For example, ``"{problem_cache_files:[cache.json]}"`` uses or fills a
       problem cache of tuned solutions.
   * - ``--gpu-arch <arch>``
     - Cross-compiles for that GPU. Kernels are not benchmarked.
   * - ``--gpu-num-cus``, ``--gpu-num-chiplets``, ``--gpu-wavefront-size``, ``--gpu-arch-params``
     - Describe the cross-compile target more precisely. They are used only with ``--gpu-arch``. See
       :ref:`cross-compilation`.
   * - ``MIGRAPHX_TRACE_BENCHMARKING=1`` (environment variable)
     - Shows the candidates and timings of each kernel.

.. note::

   For a cross-compile, any property that you do not set takes the cross-compile default, for example 120 compute
   units. The default is recorded in the output file. If the compute unit count or the chiplet count matters for
   kernel selection, set it to match the real GPU.

Step 4: Run and measure
--------------------------------------------------------------------

.. code-block:: shell

   migraphx-driver perf model_this_gpu.mxr

For a fair comparison, also compile the model directly for the new GPU and measure it:

.. code-block:: shell

   migraphx-driver compile model.onnx --gpu -o model_this_gpu_direct.mxr
   migraphx-driver perf model_this_gpu_direct.mxr

The recompiled file keeps the graph decisions that were made for the original GPU, so it can be slower than a direct
compile, especially across generations.

Step 5 (optional): Check the accuracy
--------------------------------------------------------------------

If you have the ONNX model, compare the recompiled file against the reference implementation:

.. code-block:: shell

   migraphx-driver verify --onnx model.onnx --compiled-model model_this_gpu.mxr

Pass the same input options, for example ``--input-dim``, that you used in step 1.

.. _recompile-read-output:

Understand the ``read`` output
====================================================================

A recompilable file stores, in the main module, the GPU it was compiled for and the options that it was compiled
with. Each kernel names its recipe. This is the output for a one-kernel model compiled for ``gfx1100``:

.. code-block:: text

   module: "main"
   @0 = @comment[gpu_compile_options]: {"backend_options":{},"compile_mode":"balanced","exhaustive_tune":false,"fast_math":true,"offload_copy":false,"recompilable":true} -> float_type, {}, {}
   @1 = @comment[gpu_target]: {"arch":"gfx1100","max_shared_memory_per_block":65536,"max_threads_per_block":1024,"max_threads_per_cu":2048,"num_chiplets":1,"num_cu":120,"wavefront_size":32} -> float_type, {}, {}
   ...
   @5 = gpu::code_object[code_object=4832,symbol_name=exp_kernel,global=5,local=1024,module_id=0,](x,main:#output_0), [main:recipe0] -> float_type, {10}, {1}

   module: "main:recipe0"
   main:recipe0:1 = @param:1 -> float_type, {10}, {1}
   main:recipe0:0 = @param:0 -> float_type, {10}, {1}
   main:recipe0:@2 = gpu::precompile_op[op=pointwise,additional_args=1,ignore_modules=0,output_shape=nullopt](main:recipe0:0,main:recipe0:1), [main:pointwise0] -> float_type, {10}, {1}
   main:recipe0:@3 = @return(main:recipe0:@2)

* ``@comment[gpu_target]`` is the device that the kernels were built for. Recompile updates it to the new device, so
  you can always tell what a file is compiled for.
* ``@comment[gpu_compile_options]`` records the options that the file was compiled with, so that you know how to
  compile the model the same way again. Recompile does not read or change it.
* ``[main:recipeN]`` at the end of a ``gpu::code_object`` links the kernel to its recipe. Kernels without this link,
  such as rocBLAS, MIOpen, and Composable Kernel kernels, cannot be rebuilt for another GPU.
* ``gpu::precompile_op[op=...]`` inside the recipe is the operation that the kernel was compiled from. Recompile
  compiles it again for the new GPU.
* A fused rocMLIR kernel also references its fusion module, here ``[main:pointwise0]``, which holds the operators that
  make up the kernel.
* Recipe modules never run. They are marked as bypassed, so ``run`` and ``perf`` skip them, but ``read``, ``run``, and
  ``perf`` always print them.

What happens in each situation
====================================================================

.. list-table::
   :header-rows: 1

   * - Recompile to
     - Result
   * - The same GPU that the file was compiled for
     - Every kernel with a recipe is rebuilt and tuned again. Instructions without a recipe, such as library calls,
       are kept as they are, because they are still valid on this GPU.
   * - A different GPU of the same generation, a newer one, or an older one
     - Every kernel is rebuilt for the new GPU, if all of them have a recipe. If one has not, recompile stops. Going to
       an older generation prints a warning.
   * - ``--gpu-arch <arch>``
     - Kernels are built for that GPU without benchmarking. The file runs after it is loaded on that GPU.

The generation is the major number of the ``gfx`` name. ``gfx90a`` and ``gfx942`` are generation 9, ``gfx1100`` and
``gfx1151`` are generation 11, and ``gfx1201`` is generation 12.

"The same GPU" compares all the recorded device properties: the architecture, the compute unit and chiplet counts,
the thread limits, the wavefront size, and the shared memory size. The same architecture with a different compute unit
count counts as a different GPU.

Use the C++ library API
====================================================================

This is the C++ library API that ``migraphx-driver`` uses. The public C++ API in ``migraphx.hpp``, which wraps the C
API, does not have ``recompile``.

.. code-block:: cpp

   // Compile once, recording the recipes
   migraphx::compile_options options;
   options.recompilable = true;
   p.compile(migraphx::make_target("gpu"), options);
   migraphx::save(p, "model.mxr");

   // Later, possibly on a machine without that GPU: open without loading the kernels
   migraphx::file_options fo;
   fo.finalize = false;
   auto q      = migraphx::load("model.mxr", fo);

   // Recompile for another GPU and save
   q.recompile(migraphx::make_target_from_value("gpu", migraphx::value{{"gpu_arch", "gfx942"}}));
   migraphx::save(q, "model_gfx942.mxr");

To recompile for the GPU in the machine, pass ``migraphx::make_target("gpu")``. To use exhaustive tuning, pass a second
argument with ``exhaustive_tune`` set. If ``recompile`` fails, the program is left unchanged. A target that does not
support recompile throws ``recompile: the <name> target does not support recompile``.

.. _recompile-limitations:

Limitations
====================================================================

Most of these are refused with a clear message, instead of producing a wrong result.

Kernels that cannot be rebuilt
--------------------------------------------------------------------

Only the kernels that MIGraphX generates itself, which are JIT and rocMLIR kernels, have recipes. These do not:

* rocBLAS and hipBLASLt gemms
* MIOpen convolutions
* Composable Kernel (CK) kernels
* Operations whose compile did not produce a single kernel

On the same GPU, these are kept as compiled. Each was built for that GPU, so it is still valid, and recompile keeps it
and rebuilds the other kernels. On a different GPU such an instruction cannot be reused, so any one of them stops the
whole recompile. If any remain, compile the model from the source for the new GPU.

Compiling with ``--recompilable`` avoids most of these kernels, as described in :ref:`recompile-prerequisites`, but not
all of them. A file can still have kernels with no recipe when:

* rocMLIR is not part of the build, or ``MIGRAPHX_DISABLE_MLIR`` is set.
* ``MIGRAPHX_ENABLE_CK`` is set, which selects CK kernels.
* rocMLIR does not support the operation even when it is forced, for example a dot whose output is fp8.
* The operation does not compile to a single kernel, or its inputs cannot be traced back from the kernel.

Graph decisions are not revisited
--------------------------------------------------------------------

Fusions, layouts, which library runs a gemm or convolution, the memory plan, and the stream schedule are chosen for the
original GPU and stay as they are. Across generations, a direct compile from the model can be faster.

No new memory
--------------------------------------------------------------------

The new GPU might pick a kernel configuration that needs a scratch buffer that the original memory plan does not have.
Such a recompile is refused.

Devices
--------------------------------------------------------------------

* Moving to an older generation is allowed and prints a warning. A kernel that uses a feature that the older GPU lacks
  fails while it is compiled for that GPU. The input program is left unchanged.
* Data types are not checked up front. A file that uses a type that the new GPU does not support, for example fp8 FNUZ
  on ``gfx11``, may fail later while compiling or running.
* A cross-compile does not benchmark. It takes the first valid configuration of each kernel, so tune the file again on
  the real GPU for the best performance.

Targets and APIs
--------------------------------------------------------------------

* GPU target only, and only programs that were compiled for a single target.
* ``migraphx-driver`` and the C++ library API only. There is no C API, Python API, or public C++ wrapper API.
* The compile options recorded in the file are not reused. Pass ``--exhaustive-tune`` and ``--backend-options`` again on
  each recompile where you want them.

Files
--------------------------------------------------------------------

* Only files compiled with ``--recompilable`` by the same build can be recompiled. Older ``.mxr`` files must be compiled
  again from the model.
* ``--recompilable`` makes files somewhat larger: one small recipe module per kernel, plus the fusion modules that it
  references.

Troubleshoot
====================================================================

.. list-table::
   :header-rows: 1

   * - Message (shortened)
     - What to do
   * - ``the program does not record the device it was compiled for``
     - The file was compiled without ``--recompilable``, or by an older build. Compile the model again with
       ``--recompilable``.
   * - ``the program was not compiled to be recompilable``
     - Same as above.
   * - ``a recipe parameter of ... is not a path to its inputs``
     - Same as above.
   * - Warning: ``retargeting from gfx11 (gfx1100) to the older gfx9 (gfx942); kernels that use features the older arch
       lacks may fail to compile``
     - Not an error. If a kernel then fails to compile, compile the model from the source for that GPU.
   * - ``cannot retarget to gfxXXXX: these instructions ... cannot be regenerated (CK, hipBLASLt / rocBLAS, MIOpen, or
       kernels without a recipe)``
     - The model uses library kernels. Compile it from the model for the new GPU.
   * - ``the recompiled kernels need memory that the compiled program does not have``
     - Compile the model again for the new GPU.
   * - ``an output file is required (-o <out.mxr>)``
     - Add ``-o <out.mxr>``.
   * - ``... is not a compiled program``
     - The input is not a compiled ``.mxr`` file.
   * - ``the program was not compiled for the <name> target``
     - The file was compiled for a different target than the one that you passed to ``recompile``. Use ``--gpu``.
   * - ``the <name> target does not support recompile``
     - Recompile supports the GPU target only. Use ``--gpu``.
