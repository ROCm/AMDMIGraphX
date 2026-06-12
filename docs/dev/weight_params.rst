.. meta::
  :description: MIGraphX external weights as parameters (MXR weight baking) reference
  :keywords: MIGraphX, external weights, weight baking, MXR, parameters, create_program_with_weights

.. _weight-params:

External Weights as Parameters (MXR Weight Baking)
==================================================

Large ONNX models often store their initializers (weights) in separate
external-data files rather than inside the ``.onnx`` protobuf. By default the
ONNX parser reads those files and bakes the values directly into the program as
``@literal`` instructions. Changing the weights then requires re-parsing and
re-compiling the whole model.

The *external weights as parameters* workflow keeps the weights out of the
graph at parse time by turning each external-data initializer into a
``@param`` instruction. This lets you:

1. **Parse once** -- no weight file I/O at parse time.
2. **Compile once** -- shapes are known, only the values are deferred.
3. **Save the compiled template** -- reuse it without re-parse/re-compile.
4. **Bake a weight set** -- :cpp:func:`create_program_with_weights` reads raw
   weight bytes from a directory and produces a new, self-contained program.
5. **Save the baked program as an MXR** -- deploy a model with the weights
   built in, or stamp out many MXRs (one per weight set) from a single template.

Because step 4 only swaps the deferred parameter values for literals, multiple
weight sets can be baked from one compiled template without any recompilation.

Workflow Overview
-----------------

.. code-block:: text

    parse_onnx(external_weights_as_parameters=True)   # weights -> @param
                   |
                   v
    program.compile(target)                           # compile template once
                   |
                   v
    create_program_with_weights(prog, dir, target)    # @param -> baked @literal
                   |                                     (per weight set)
                   v
    save(baked, "model.mxr")                          # serialize self-contained MXR
                   |
                   v
    load("model.mxr")                                 # finalizes on load, ready to run

Enabling the Workflow
---------------------

The parser only emits weight parameters when the
``external_weights_as_parameters`` option is set. When enabled, every
external-data initializer becomes a ``@param`` and its file metadata
(location, offset, length) is recorded in the program's *external weight map*.
:cpp:func:`create_program_with_weights` later uses that map to locate the raw
bytes for each weight.

.. note::

   Only initializers that use ONNX external data become parameters. Inline
   initializers stored inside the ``.onnx`` file are still parsed as literals.
   ``create_program_with_weights`` requires a non-empty external weight map; it
   errors if the program was not parsed with this option.

C++ API
-------

Parsing the template
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    #include <migraphx/migraphx.hpp>

    migraphx::onnx_options options;
    options.set_external_weights_as_parameters(true);
    auto prog = migraphx::parse_onnx("model.onnx", options);

    migraphx::target t = migraphx::target("gpu");
    prog.compile(t);

Baking a weight set
^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // Copy the compiled template and replace external-weight parameters with
    // literals read from base_dir, lowering them for the given target.
    migraphx::program baked =
        migraphx::create_program_with_weights(prog, "weights_v1", t);

    // Persist the self-contained result.
    migraphx::file_options fo;
    migraphx::save(baked, "model_v1.mxr", fo);

The free function is declared in ``migraphx/program.hpp``:

.. code-block:: cpp

    /// Copy the program and replace external-weight parameters with literals
    /// read from base_dir, producing a self-contained program suitable for
    /// saving as an MXR. The target is used to lower the baked literals for the
    /// device (the equivalent of write_literals + finalize).
    MIGRAPHX_EXPORT program create_program_with_weights(const program& prog,
                                                        const std::string& base_dir,
                                                        const target& t);

The ``target`` argument controls how the baked literals are lowered. For GPU
targets the literals are lowered to the device representation (``gpu::literal``)
so the baked program is ready to run on the device.

Running a baked program in-process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``create_program_with_weights`` deliberately does **not** finalize the program.
Finalizing uploads literal data to the device, which is wasted work if you only
intend to ``save`` the result (the bytes would be serialized after a redundant
host-to-device round trip).

The baked program therefore is not yet runnable on the device. The portable way
to make it runnable is to save it and load it back: loading a compiled MXR
finalizes it automatically, allocating device buffers and uploading the baked
literals.

.. code-block:: cpp

    auto baked = migraphx::create_program_with_weights(prog, "weights_v1", t);
    migraphx::save(baked, "model_v1.mxr");        // serialize the baked program

    auto runnable = migraphx::load("model_v1.mxr"); // finalized on load
    auto outputs  = runnable.eval(params);

.. note::

   The underlying core library has a ``program::finalize(const target&)`` method
   that finalizes a baked or loaded program in place, but it is **not** exposed
   through the C or C++ API wrappers (``migraphx.h`` / ``migraphx.hpp``). From
   C++, use the save/load round trip above. From Python the method *is* exposed
   (see below) if you want to avoid touching disk.

Python API
----------

The same workflow is available through the Python bindings.

.. code-block:: python

    import migraphx
    import numpy as np

    # 1. Parse with weights kept as parameters.
    prog = migraphx.parse_onnx("model.onnx",
                               external_weights_as_parameters=True)

    # The external weights show up as ordinary parameters.
    params = prog.get_parameter_shapes()
    assert "W" in params

    # 2. Compile the template once.
    gpu = migraphx.get_target("gpu")
    prog.compile(gpu, offload_copy=False)

    # 3. Bake a weight set from a directory of raw .bin files.
    baked = migraphx.create_program_with_weights(prog, "weights_v1", gpu)
    assert "W" not in baked.get_parameter_shapes()   # weight is now a literal

    # 4. Save the self-contained MXR for later deployment.
    migraphx.save(baked, "model_v1.mxr")

    # 5. Run in-process. Baking does not finalize, so materialize the device
    #    buffers first. Python exposes program.finalize() directly...
    baked.finalize(gpu)
    # ...or, equivalently and portably, reload the saved MXR which finalizes
    #    on load:  runnable = migraphx.load("model_v1.mxr")

    run_params = {}
    for name, shape in baked.get_parameter_shapes().items():
        if name == "input":
            x = np.ones([1, 4], dtype=np.float32)
            run_params[name] = migraphx.to_gpu(migraphx.argument(x))
        else:
            run_params[name] = migraphx.allocate_gpu(shape)
    result = baked.run(run_params)

``parse_onnx`` and ``parse_onnx_buffer`` both accept the
``external_weights_as_parameters`` keyword (default ``False``), and
``create_program_with_weights(program, base_dir, target)`` mirrors the C++
function above. Unlike the C/C++ wrappers, the Python bindings additionally
expose ``program.finalize(t)``, which finalizes a baked program in place so it
can be run without a disk round trip.

CLI
---

The ``migraphx-driver`` tool exposes the template-parsing step with the
``--weight-params`` flag. It sets ``external_weights_as_parameters`` so the
parsed/compiled program keeps its external weights as parameters. Combined with
``compile -o`` this is how you build and save a reusable template ``.mxr``:

.. code-block:: bash

    migraphx-driver read model.onnx --weight-params
    migraphx-driver compile model.onnx --weight-params --gpu -o template.mxr

The ``compile`` command can also bake a weight set in the same step with
``--bake-weights <dir>``. The input may be the ONNX model directly or a
previously-saved template ``.mxr`` (which already carries the external weight
map); the result is a self-contained program written to ``-o``:

.. code-block:: bash

    # Bake straight from the ONNX model.
    migraphx-driver compile model.onnx --weight-params --gpu \
        --bake-weights weights_v1 -o model_v1.mxr

    # ...or stamp a weight set into an existing compiled template.
    migraphx-driver compile template.mxr --gpu \
        --bake-weights weights_v1 -o model_v1.mxr

When the input is an already-compiled template, the driver skips compilation,
bakes the weights from ``<dir>``, and saves the result. ``--bake-weights``
requires that the program was parsed with ``--weight-params`` (so it has a
non-empty external weight map); otherwise the driver errors.

Weight Directory Layout
-----------------------

``create_program_with_weights`` reads each weight from ``base_dir`` using the
``location``, ``offset`` and ``length`` recorded in the external weight map when
the model was parsed. In practice this means ``base_dir`` must contain the same
external-data ``.bin`` file(s) the ONNX model refers to, with the raw tensor
bytes laid out exactly as the model expects. To bake a different weight set,
point ``base_dir`` at a directory containing differently-valued ``.bin`` files
with the same layout.

Notes and Caveats
-----------------

* ``create_program_with_weights`` operates on a *compiled* program; the returned
  program is a copy, leaving the template untouched so it can be reused for the
  next weight set.
* The function errors if the program has no external weight map -- always parse
  with ``external_weights_as_parameters`` enabled before baking.
* Baking does not finalize. To run a baked program in-process, save it and load
  it back -- loading a compiled MXR finalizes it automatically. (Python also
  exposes ``program.finalize(target)`` as an in-place alternative; the C/C++ API
  wrappers do not.)
* When compiling with ``offload_copy=False`` you must provide device buffers for
  every remaining parameter (including output buffers) at run time, as shown in
  the Python example.
* Working end-to-end examples live under
  ``examples/migraphx/weight_params/`` (for example ``gpu_bake_test.py`` and
  ``resnet50_weight_baking.py``).
