.. meta::
  :description: Specializing a symbolic dimension at compile time in MIGraphX
  :keywords: MIGraphX, code base, contribution, developing, symbolic shapes, dynamic shapes, split_single_dyn_dim, select_module

Specializing a symbolic dimension: design and review guide
===========================================================

Purpose
-------

KV-cache language models use the same exported ONNX graph in two materially different ways:

* **Decode** processes one new token.
* **Prefill** processes a prompt, padded to the configured maximum sequence length.

Compiling two independent programs works, but duplicates model state and requires the caller to
load and manage two programs. Leaving the sequence length fully dynamic avoids that, but gives up
the shape information that kernel selection and fusion depend on.

Endpoint specialization is the middle path. One MIGraphX program holds a static specialization per
size the model is actually run at. The main module keeps the dynamic interface and dispatches to
the matching specialization from the concrete runtime input shapes.

At a glance
-----------

Given an ONNX input parsed with a symbolic sequence length::

  x: [1, sequence_length, hidden]
  sequence_length: {min: 1, max: MAX_SEQ_LEN}

compiling with ``split_sizes = {1, MAX_SEQ_LEN}`` produces::

                           main
                  dynamic inputs and literals
                             |
                select_module(input shapes)
                   /                     \
        specialization                specialization
      sequence_length = 1           sequence_length = MAX_SEQ_LEN
                   \                     /
                    tuple of model outputs
                             |
              get_tuple_elem for each ONNX output

Only the requested sizes are supported. A runtime sequence length that was not named does not
match any specialization, and evaluation reports
``SELECT_MODULE: no compatible submodules found`` rather than running a specialization built for
a different length. Prefill callers are expected to pad to the configured maximum.

User-facing contract
--------------------

Specialization is a compile-time transformation, driven by ``compile_options::split_sizes`` and
performed by the ``split_single_dyn_dim`` pass. Nothing about it involves the ONNX parser: the
model is parsed once, against a symbolic dimension, exactly as any other symbolic model is.

Two things are required:

* The model is parsed with ``use_symbolic_shapes = true`` and a ``dim_params`` entry giving the
  dimension its range. The pass identifies the axis by its symbol, so an ordinary bounded dynamic
  dimension carries nothing for it to key on.
* The target implements specialization. Only the GPU target does, in its dynamic-shapes pipeline.
  Targets that do not, reject a non-empty ``split_sizes`` through ``throw_if_split_sizes_set``
  rather than returning a program that runs but was never specialized.

Leaving ``split_sizes`` empty is valid and means every size the dimension can take, which is only
reasonable for a narrow range. A model run at a few known sizes should name them.

The parse/compile side of a driver workflow is::

  migraphx-driver compile model.onnx --gpu \
      --enable-symbolic \
      --split-sizes 1 2048 \
      --dim-param "@sequence_length" "{min:1, max:2048}" \
      -o model.mxr

When running a saved program, a fixed ``--dim-param`` value binds the symbol used by generated
inputs and output allocations. Use ``1`` for decode or ``MAX_SEQ_LEN`` for prefill::

  migraphx-driver run model.mxr --migraphx \
      --dim-param "@sequence_length" 1

Python exposes the same option on ``compile``::

  program = migraphx.parse_onnx(
      "model.onnx",
      use_symbolic_shapes=True,
      dim_params={"sequence_length": migraphx.shape.dynamic_dimension(1, 2048)})
  program.compile(migraphx.get_target("gpu"), split_sizes=[1, 2048])

Reach
-----

Endpoint specialization only reaches models that parse symbolically in the first place. Many
operator parsers read concrete lengths while building IR: 48 of the 105 ``parse_*.cpp`` files call
``shape::lens()``, which throws on a dynamic input.

For the language models this targets, that population no longer includes the operators that
matter. ``GroupQueryAttention`` and rotary embedding parse symbolically, as does ``MatMulNBits``,
so quantized exports are reachable. ``Attention`` and ``MultiHeadAttention`` still read concrete
lengths and therefore cannot be specialized this way; models exported with those operators need a
statically parsed program per phase.

Sliding-window ``GroupQueryAttention`` used to be the clearest counterexample, since it rejected a
symbolic sequence length outright. That guard existed because the two phases were given window
bounds that differed by one key, so the bound could not be written without knowing the phase.
Checking the intended bound against onnxruntime showed that both bounds were wrong and that the
correct one, ``local_window_size - 1`` keys before the row's absolute cache position, does not
depend on the phase at all. With the branch gone the operator parses symbolically like any other.

Main-module output shapes
-------------------------

The specializations have static output shapes, but ``select_module`` needs one shape description
that covers any of them:

* Element types and ranks agree across specializations.
* Equal dimensions remain static.
* A dimension that follows the specialized axis keeps its symbol, so a caller sizing an output
  buffer resolves it exactly as it resolves the inputs.
* A dimension that varies for some other reason falls back to an ordinary min/max dynamic range.

The symbolic form is important for allocation. The selected submodule writes into output storage
owned by the main module. Before execution, ``select_module`` reshapes each sufficiently large
output buffer to the selected module's exact static output shape. It rejects undersized buffers
instead of relying on an assertion or permitting an out-of-bounds write.

``simplify_dyn_ops`` does not replace a symbolic ``select_module`` output with a less precise
plain range recovered from the static submodules.

Literal capture
---------------

Initializers are deliberately shared. They remain main-module literals captured by the
specializations instead of being duplicated into each one or exposed as runtime parameters.
Constants created inside an individual operator parser remain local to that specialization.

All specializations expose the same parameter names and number of outputs. Inputs are passed in
sorted parameter-name order, matching the existing ``select_module`` convention.

Compilation and runtime dispatch
--------------------------------

After compilation, the main module is intentionally a thin wrapper:

* runtime inputs and shared leaf captures;
* a tuple output allocation;
* ``select_module``;
* one ``get_tuple_elem`` per model output; and
* ``@return``.

``select_module`` caches metadata for each candidate module set. The metadata records:

* input and output parameter names and shapes;
* each compiled parameter's positional order;
* the input indices whose names or shapes differ between candidates; and
* whether every value captured from outside the candidates is a leaf.

At runtime, only the discriminating input indices are compared to choose a candidate. The chosen
module still performs normal parameter-shape validation during evaluation.

The evaluator supports a positional parameter view in addition to the normal string-keyed
``parameter_map``. This lets ``select_module`` route existing input arguments and tuple output
subobjects directly to the selected module by compiled parameter order, avoiding construction and
lookup of a per-evaluation string map.

Captured initializers require one further evaluator behaviour: when an instruction references a
foreign instruction that is not already in the current result map, the evaluator may compute it
lazily only if it is a leaf. Literals and context-free/context-bound zero-input operations are
supported; a non-leaf foreign dependency is rejected.

Evaluation fast path
--------------------

Normal, untraced evaluation recognizes the thin wrapper shape described above. If all captures
are leaves, it:

1. evaluates only the main-module parameters and captures needed by ``select_module``;
2. selects and evaluates one specialization;
3. returns the requested tuple subobjects directly.

This avoids populating a result map for the wrapper and avoids evaluating the chain of
``get_tuple_elem`` instructions. The generic path remains available for other graph shapes and
for tracing, preserving instruction-level trace behavior.

Other overhead reductions support this path:

* ``argument::get_sub_object(index)`` returns one tuple member directly, instead of constructing
  the full vector returned by ``get_sub_objects()``.
* ``get_tuple_elem`` uses that direct accessor.
* ``select_module`` metadata is built once and reused. Cache lookup has an atomic last-entry fast
  path and a mutex-protected fallback, so concurrent evaluation does not race metadata creation.

Allocation behavior
-------------------

Compiled ``select_module`` aliases its tuple output allocation. Each following
``get_tuple_elem`` aliases one subobject of that tuple.

``replace_allocate`` previously saw the whole tuple allocation while traversing aliases and could
insert a redundant copy for each returned tuple member. It now recognizes a
``get_tuple_elem`` whose shape matches the corresponding subobject of an aliased tuple allocation
and leaves that caller-owned storage in place.

This is specific to a verified tuple-subobject relationship; ordinary tuple results that do not
alias an allocation retain the existing copy behavior.

Supporting changes
------------------

Attention submodule names
~~~~~~~~~~~~~~~~~~~~~~~~~

``fuse_attention`` runs inside each specialization. Its local numbering restarted for each parent
and could create duplicate program-wide names such as ``attn0``. Non-main attention submodules are
qualified with the parent module name; main-module names remain unchanged.

Driver argument generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The driver turns fixed ``--dim-param`` values into symbolic bindings when generating arguments.
This is needed because the existing ``--batch`` fallback cannot identify an arbitrary symbolic axis
such as ``sequence_length``.

Tuple parameter shapes are resolved recursively. If a symbolic input cannot be fully resolved,
the driver reports which symbols were bound. Tuple output allocations are also excluded from the
warning about integral user inputs.

rocMLIR compatibility guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU CMake configuration probes for MIGraphX dialect C API version ``6`` before defining
``MIGRAPHX_MLIR`` and linking rocMLIR.

The existing translation-unit guard in ``mlir.cpp`` was too late: other passes could still build
``mlir_op`` modules even though no compatible compiler was available to compile them. An
incompatible rocMLIR installation now produces a configure warning and builds with MLIR support
disabled consistently. The probe result is deliberately removed from the CMake cache before each
check so an in-place rocMLIR upgrade is detected.

Review map
----------

The main files are:

* ``src/split_single_dyn_dim.cpp``: submodule construction per requested size, literal capture,
  and main-module rewriting.
* ``src/include/migraphx/compile_options.hpp``: the ``split_sizes`` option and the rejection
  helper for targets that cannot honour it.
* ``src/targets/gpu/target.cpp``: where the pass is run in the dynamic-shapes pipeline.
* ``src/include/migraphx/op/select_module.hpp``: candidate metadata, shape-based selection,
  positional parameter routing, output-buffer preparation, and metadata caching.
* ``src/program.cpp``: positional evaluation, leaf captures, and the whole-wrapper fast path.
* ``src/replace_allocate.cpp``, ``src/argument.cpp``, and ``get_tuple_elem.hpp``: tuple output
  aliasing and per-evaluation overhead reductions.
* ``src/driver/main.cpp``: the ``--split-sizes`` option and symbolic argument generation.
* ``src/fuse_attention.cpp`` and ``src/simplify_dyn_ops.cpp``: compilation fixes exposed by
  compiling nested static graphs.

Tests and intended coverage
---------------------------

``test/onnx/verify/split_symbolic_test.cpp`` covers, on the reference target:

* construction of both specializations and their static parameter shapes;
* initializer sharing, with the literals staying in the main module;
* preservation of an independent symbolic dimension;
* decode and prefill results against the same model parsed at that fixed length;
* rejection of a sequence length that was not specialized;
* serialization and reload of a compiled, specialized program; and
* ``GroupQueryAttention`` structure and results.

``test/gpu/split_symbolic.cpp`` compares both endpoint phases with reference results on a ROCm
device, driving the split through ``compile_options::split_sizes``.

``test/compile_options.cpp`` and ``test/targets.cpp`` cover the option default and the rejection
by targets that cannot specialize, both directly and through ``program::compile``.

``test/ref/select_module.cpp`` covers dispatch after save/load, captured literals, ordinary and
traced evaluation, and multiple candidate shapes.

``test/argument_test.cpp`` and ``test/replace_allocate.cpp`` cover direct tuple access and the
distinction between aliased allocation subobjects and ordinary tuple outputs.

``test_driver_symbolic_args`` exercises compile-then-run driver commands with fixed decode and
prefill symbol bindings. ``test/simplify_dyn_ops_test.cpp`` directly verifies that the symbolic
output shape is preserved. ``test/py/test_split_sizes.py`` covers the Python compile option,
including its rejection by the reference target.

The GPU test deliberately uses the small deterministic Add model. ``GroupQueryAttention``
numerical coverage remains on the reference target because its broader GPU test path has separate
known constraints.

Comparing two compiled programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diffing the printed IR of two compiled programs does not work, and quietly gives a misleading
answer rather than an obviously wrong one. Literal ordering is not deterministic: one binary
compiling one model twice produced IR differing in 894 lines on SmolVLM2-500M, with no change to
anything. Compare normalized per-module instruction histograms instead, and validate whichever
comparison is used by first running it on two artifacts built by the same binary. That control is
what distinguishes a real difference from ordering noise.

Latency comparisons need the same care. Interleave the runs rather than measuring one artifact and
then the other, and include a second artifact built by the same binary as a third arm, since two
artifacts differing only by ordering can differ in latency by as much as a genuine change would.

Benchmarking these models at all needs ``--fill1`` on the integer inputs. The driver otherwise
fills parameters randomly, and a random ``int64`` attention mask becomes an out-of-range index that
faults ``gathernd``. A fault leaves ``gpucore.*.gpu`` dumps in the working directory.

Boundaries and non-goals
------------------------

* This is endpoint specialization, not general dynamic-shape compilation. Sizes that were not
  requested are intentionally unsupported.
* Specialization is implemented by the GPU target only. Other targets reject the request.
* The specializations are compiled independently inside one program; this shares initializers and
  runtime ownership, not generated kernels.
* Candidate selection is based on exact concrete input shapes. The first matching candidate wins.
* The fast evaluator path is an optimization only. Graphs that do not match its structural and
  capture-safety checks use the generic evaluator.
