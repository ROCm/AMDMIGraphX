.. meta::
  :description: How GPU cross-compilation works in MIGraphX
  :keywords: MIGraphX, cross-compile, cross-compilation, gpu, gfx, mxr

.. _cross-compilation:

Cross-Compilation
=================

Cross-compilation lets MIGraphX compile a model for a GPU architecture that is
not present on the build machine. The compiler produces a serialized program
(``.mxr``) targeting a requested ``gfx`` architecture without a physical device,
allocating GPU memory, or running any kernels. The resulting ``.mxr`` is later
loaded and finalized on a machine that has the matching device, where the
remaining device-side work (allocating and preparing device buffers) is performed.
NOTE: Only supports rocMLIR and MIGraphX exclusive kernels.

Using cross-compilation
-----------------------

With migraphx-driver
^^^^^^^^^^^^^^^^^^^^^

Pass ``--gpu-arch`` for a quick way to cross-compile for a specific architecture
without any advanced device configuration:

.. code-block:: bash

    migraphx-driver compile model.onnx --gpu-arch gfx942 -o model_gfx942.mxr

For specific device configurations, use
``--gpu-arch-params``, a JSON object that is only used when ``--gpu-arch`` is set.
It accepts ``num_cu``, ``num_chiplets``, ``max_threads_per_cu``,
``max_threads_per_block``, and ``wavefront_size``. For example, to cross-compile
for a ``gfx942`` with 304 compute units and 8 chiplets:

.. code-block:: bash

    migraphx-driver compile model.onnx --gpu-arch gfx942 \
        --gpu-arch-params "{num_cu:304, num_chiplets:8}" -o model_gfx942.mxr

Set ``wavefront_size`` explicitly to override the inferred value. The default
``0`` infers wavefront size from the architecture name (wave32 for RDNA
``gfx10``/``gfx11``/``gfx12``, wave64 otherwise). Use ``--gpu-wavefront-size`` or
``wavefront_size`` in ``--gpu-arch-params`` to override:

.. code-block:: bash

    migraphx-driver compile model.onnx --gpu-arch gfx942 \
        --gpu-arch-params "{wavefront_size:32, num_cu:60}" -o model_gfx942.mxr

The produced ``.mxr`` is loaded on the target machine like any other compiled
program:

.. code-block:: bash

    migraphx-driver run model_gfx942.mxr --migraphx

With the library API
^^^^^^^^^^^^^^^^^^^^

Construct the ``gpu`` target with the cross-compile fields set and compile as
usual:

.. code-block:: cpp

    migraphx::target t =
        migraphx::make_target("gpu", migraphx::value{{"gpu_arch", "gfx942"}});
    p.compile(t);

    // Serialize for later use on a gfx942 device.
    migraphx::save(p, "model_gfx942.mxr");

The recognized fields (with defaults) are ``gpu_arch`` (empty), ``gpu_num_cu``
(``120``), ``gpu_num_chiplets`` (``1``), ``gpu_max_threads_per_cu`` (``2048``),
``gpu_max_threads_per_block`` (``1024``), and ``gpu_wavefront_size`` (``0``).
A value of ``0`` for ``gpu_wavefront_size`` infers wavefront size from the
architecture; set ``32`` or ``64`` explicitly to override.
A non-empty ``gpu_arch`` is what puts the target into cross-compile mode.

.. code-block:: cpp

    migraphx::target t = migraphx::make_target(
        "gpu", migraphx::value{{"gpu_arch", "gfx942"}, {"gpu_wavefront_size", 32}});

How it works
------------

The cross-compile target and context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GPU ``target`` (``src/targets/gpu/include/migraphx/gpu/target.hpp``) stores
the requested architecture and synthetic device parameters. It reports
cross-compile mode when an architecture is set:

.. code-block:: cpp

    bool is_cross_compile() const { return not gpu_arch.empty(); }

In cross-compile mode, ``target::get_context()`` builds a context backed by a
synthetic ``hipDeviceProp_t`` instead of querying a real device. The synthetic
properties are filled in by ``make_cross_compile_device_props``
(``src/targets/gpu/cross_compile_device.cpp``), which sets the arch name, compute
unit count, chiplet count, max threads, and wavefront size. When
``gpu_wavefront_size`` is ``0``, wavefront size is inferred from the architecture
(wave32 for RDNA ``gfx10``/``gfx11``/``gfx12``, wave64 otherwise); otherwise the
explicit ``32`` or ``64`` override is used.

A cross-compile context cannot touch a device. The target's ``copy_to``,
``copy_from``, and ``allocate`` all throw in this mode, and the context's
``finish``, ``get_queue``, and ``wait_for`` throw as well. This guarantees that
nothing in the pipeline silently depends on a real device being present.

Compilation pipeline
^^^^^^^^^^^^^^^^^^^^

Cross-compilation runs the same pass pipeline as a native GPU compile
(``target::get_passes``); kernels are still generated as code objects for the
requested architecture. Two things change:

* **Kernel benchmarking is skipped.** ``compile_ops`` does not benchmark when the
  context is in cross-compile mode (``ctx->is_cross_compile()``), since
  benchmarking would require running kernels on a device. The first/default
  solution is selected instead, so ``exhaustive_tune`` has no effect.
* **Finalization is skipped.** ``program::compile`` only finalizes the modules
  when the context is not cross-compiling. Finalization is where device-side work
  happens, so deferring it keeps the compile device-free.

Loading and running a cross-compiled program
--------------------------------------------

When a ``.mxr`` is loaded with ``program::from_value`` on the deployment
machine, the GPU target is reconstructed from the serialized value, but the
context is rebuilt against the **real** local device: ``context::from_value``
replaces the device with one created from ``get_device_id()``. The loaded
program is therefore no longer cross-compiling.

``from_value`` then calls ``program::finalize``, which performs the device-side
work that was deferred during cross-compilation: it runs the target's finalize
passes and then the per-instruction ``finalize`` step, allocating device buffers
and preparing each op to run. After this the program is fully finalized and can
be run like a natively compiled one.

Limitations and caveats
-----------------------

* **The architecture must match at load time.** Code objects are compiled for
  the requested ``gfx`` architecture. Loading a ``.mxr`` on a device with a
  different architecture fails when the code objects are loaded.
* **Only rocMLIR and MIGraphX device kernels are supported.** Cross-compilation
  can only produce kernels that are emitted as code objects without a device.
  Library backends (MIOpen, rocBLAS, hipBLASLt, and Composable Kernel) require a
  physical device and are not supported. ``MIGRAPHX_DISABLE_MLIR``,
  ``MIGRAPHX_ENABLE_CK``, ``MIGRAPHX_SET_GEMM_PROVIDER``, and
  ``MIGRAPHX_ENABLE_MIOPEN_POOLING`` must be unset.

* **No tuning.** Kernel benchmarking is skipped, so the selected kernels are not
  tuned for the target and performance may be lower than a native, tuned
  compile. ``exhaustive_tune`` is ineffective in this mode.
* **Synthetic device properties.** Compute unit count, chiplet count, max
  threads, and wavefront size come from the supplied parameters (or defaults).
  Inaccurate values can change kernel selection and occupancy heuristics, so set
  ``--gpu-arch-params`` to match the real device when it matters.
* **No memory validation.** Device buffer sizes are not checked against the
  target's available memory during cross-compile, so an out-of-memory condition
  can surface only at load/finalize time on the device.
* **Device operations are unavailable.** Evaluating the program, or calling
  ``copy_to``/``copy_from``/``allocate`` on a cross-compile target, throws.
