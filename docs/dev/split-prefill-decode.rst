Split prefill/decode: design and review guide
================================================

Purpose
-------

KV-cache language models use the same exported ONNX graph in two materially different ways:

* **Decode** processes one new token.
* **Prefill** processes a prompt, padded to the configured maximum sequence length.

Treating the sequence length as one ordinary dynamic dimension is not sufficient for these
models. Several operators specialize behavior or attributes from their input lengths while they
are parsed, and some parsers (notably GroupQueryAttention) cannot accept a symbolic sequence
length. Compiling two independent programs works, but duplicates model state and requires the
caller to load and manage two programs.

This change makes one MIGraphX program contain both static specializations. The main module keeps
the dynamic interface and dispatches to the matching specialization from the concrete runtime
input shapes.

At a glance
-----------

Given an ONNX input such as::

  x: [1, sequence_length, hidden]
  sequence_length: {min: 1, max: MAX_SEQ_LEN}

the parser constructs::

                           main
                  dynamic inputs and literals
                             |
                select_module(input shapes)
                   /                     \
      main:split:decode             main:split:prefill
      sequence_length = 1           sequence_length = MAX_SEQ_LEN
                   \                     /
                    tuple of model outputs
                             |
              get_tuple_elem for each ONNX output

Only the two endpoint shapes are supported. A runtime sequence length other than ``1`` or
``MAX_SEQ_LEN`` does not match either specialization. Prefill callers are expected to pad to the
configured maximum.

User-facing contract
--------------------

The C++ API adds ``onnx_options::split_prefill_decode``. The driver exposes the same option as
``--split-prefill-decode``.

The split requires all of the following:

* The option is explicitly enabled. Existing parsing behavior is unchanged by default.
* ``onnx_options::dim_params["sequence_length"]`` is a range whose minimum is ``1`` and whose
  maximum is greater than ``1``.
* At least one graph input still has an ONNX axis named ``sequence_length``. A
  ``map_input_dims`` override replaces the ONNX dimension names and therefore cannot be used to
  identify the split axis.

Invalid configurations fail during parsing instead of silently producing a program that handles
only one phase.

Symbolic shapes are recommended but not required for the split. With
``use_symbolic_shapes = true``, the main module retains the ``sequence_length`` symbol, which lets
callers and driver-generated arguments resolve related input and output dimensions consistently.
Without symbolic shapes, the main module uses ordinary bounded dynamic dimensions while the two
selected modules remain static.

For example, the parse/compile side of a driver workflow is::

  migraphx-driver compile model.onnx --gpu \
      --enable-symbolic \
      --split-prefill-decode \
      --dim-param "@sequence_length" "{min:1, max:2048}" \
      -o model.mxr

When running a saved program, a fixed ``--dim-param`` value binds the symbol used by generated
inputs and output allocations. Use ``1`` for decode or ``MAX_SEQ_LEN`` for prefill::

  migraphx-driver run model.mxr --migraphx \
      --dim-param "@sequence_length" 1

The Python file and buffer parsers expose the same options::

  options = {
      "use_symbolic_shapes": True,
      "split_prefill_decode": True,
      "dim_params": {
          "sequence_length": migraphx.shape.dynamic_dimension(1, 2048),
      },
  }
  program = migraphx.parse_onnx("model.onnx", **options)

Parse-time construction
-----------------------

The implementation lives in the ONNX parser rather than in a later graph pass. By the time an
ONNX graph has become MIGraphX IR, operator parsers may already have:

* rejected an unresolved dimension;
* selected phase-specific behavior; or
* copied concrete dimensions into operation attributes.

The first branch prototype used a standalone IR pass. It was removed when this constraint became
clear; the final branch performs the split during ONNX import. This explains why the feature is
an ONNX option rather than a target pass and why no new pass remains in the final diff.

``parse_prefill_decode`` therefore performs these steps:

1. Validate the ``sequence_length`` bounds and confirm that the graph uses the named dimension.
2. Parse graph initializers once into the main module.
3. Parse the graph into ``main:split:decode`` after temporarily fixing
   ``sequence_length = 1``.
4. Parse it again into ``main:split:prefill`` after fixing
   ``sequence_length = MAX_SEQ_LEN``.
5. Restore the caller's original dimension range and parser state.
6. Add one ``select_module`` to the main module and unpack its tuple result into the original
   ONNX outputs.

Initializers are deliberately shared. They remain main-module literals captured by both
specializations instead of being duplicated or exposed as runtime parameters. Constants created
inside an individual operator parser remain local to that specialization.

Both specializations must expose the same parameter names and number of outputs. Inputs are
passed in sorted parameter-name order, matching the existing ``select_module`` convention.

Main-module output shapes
-------------------------

The two specializations have static output shapes, but ``select_module`` needs one shape
description that covers either result. Corresponding decode and prefill outputs are combined as
follows:

* Element types and ranks must agree.
* Equal dimensions remain static.
* Dimensions spanning exactly ``[1, MAX_SEQ_LEN]`` keep the ``sequence_length`` symbol when
  symbolic parsing is enabled.
* If an output varies for some other reason, it falls back to an ordinary min/max dynamic range.

The symbolic form is important for allocation. The selected submodule writes into output storage
owned by the main module. Before execution, ``select_module`` reshapes each sufficiently large
output buffer to the selected module's exact static output shape. It rejects undersized buffers
instead of relying on an assertion or permitting an out-of-bounds write.

``simplify_dyn_ops`` no longer replaces a symbolic ``select_module`` output with a less precise
plain range recovered from the static submodules.

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
module still performs normal parameter-shape validation during evaluation. If no candidate
matches, evaluation reports ``SELECT_MODULE: no compatible submodules found``.

The evaluator now supports a positional parameter view in addition to the normal string-keyed
``parameter_map``. This lets ``select_module`` route existing input arguments and tuple output
subobjects directly to the selected module by compiled parameter order, avoiding construction and
lookup of a per-evaluation string map.

Captured initializers require one further evaluator change: when an instruction references a
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

``fuse_attention`` is now run inside both selected modules. Its local numbering restarted for
each parent and could create duplicate program-wide names such as ``attn0``. Non-main attention
submodules are now qualified with the parent module name; main-module names remain unchanged.

Driver argument generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The driver now turns fixed ``--dim-param`` values into symbolic bindings when generating
arguments. This is needed because the existing ``--batch`` fallback cannot identify an arbitrary
symbolic axis such as ``sequence_length``.

Tuple parameter shapes are resolved recursively. If a symbolic input cannot be fully resolved,
the driver reports which symbols were bound. Tuple output allocations are also excluded from the
warning about integral user inputs.

rocMLIR compatibility guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU CMake configuration now probes for MIGraphX dialect C API version ``6`` before defining
``MIGRAPHX_MLIR`` and linking rocMLIR.

The existing translation-unit guard in ``mlir.cpp`` was too late: other passes could still build
``mlir_op`` modules even though no compatible compiler was available to compile them. An
incompatible rocMLIR installation now produces a configure warning and builds with MLIR support
disabled consistently. The probe result is deliberately removed from the CMake cache before each
check so an in-place rocMLIR upgrade is detected.

Review map
----------

The main files are:

* ``src/onnx/onnx_parser.cpp``: validation, double parsing, initializer sharing, output-shape
  unification, and main-module construction.
* ``src/include/migraphx/onnx.hpp``, ``src/onnx/onnx.cpp``, and
  ``src/onnx/include/migraphx/onnx/onnx_parser.hpp``: public-to-internal option plumbing and the
  reusable-initializer parser interface.
* ``src/include/migraphx/op/select_module.hpp``: candidate metadata, shape-based selection,
  positional parameter routing, output-buffer preparation, and metadata caching.
* ``src/program.cpp``: positional evaluation, leaf captures, and the whole-wrapper fast path.
* ``src/replace_allocate.cpp``, ``src/argument.cpp``, and ``get_tuple_elem.hpp``: tuple output
  aliasing and per-evaluation overhead reductions.
* ``src/driver/main.cpp``: CLI option and symbolic argument generation.
* ``src/fuse_attention.cpp``, ``src/simplify_dyn_ops.cpp``, and
  ``src/targets/gpu/CMakeLists.txt``: compilation fixes exposed by compiling two nested static
  graphs.

Tests and intended coverage
---------------------------

``test/onnx/parse/split_prefill_decode_test.cpp`` checks:

* construction and naming of both static specializations;
* initializer sharing;
* GroupQueryAttention parsing with a formerly unsupported symbolic length;
* operation with and without symbolic shapes;
* preservation of an independent symbolic dimension;
* opt-in behavior; and
* errors for missing/invalid bounds and overridden input dimensions.

``test/onnx/verify/split_prefill_decode_test.cpp`` compiles one program for the reference target
and covers:

* one-token decode and maximum-length prefill results;
* rejection of intermediate sequence lengths;
* split-program serialization and reload;
* GroupQueryAttention results compared with fixed, unsplit specializations; and
* numerical multi-input/multi-output execution with an independent dynamic dimension.

``test/ref/select_module.cpp`` covers dispatch after save/load, captured literals, ordinary and
traced evaluation, and multiple candidate shapes.

``test/argument_test.cpp`` and ``test/replace_allocate.cpp`` cover direct tuple access and the
distinction between aliased allocation subobjects and ordinary tuple outputs.

``test/gpu/split_prefill_decode.cpp`` compares both endpoint phases with reference results on a
ROCm device. ``test_driver_symbolic_args`` exercises compile-then-run driver commands with fixed
decode and prefill symbol bindings. ``test/simplify_dyn_ops_test.cpp`` directly verifies that the
symbolic output shape is preserved. The Python tests cover both file and buffer parsing.

The GPU test deliberately uses the small deterministic Add model. GroupQueryAttention numerical
coverage remains on the reference target because its broader GPU test path has separate known
constraints.

Boundaries and non-goals
------------------------

* This is endpoint specialization, not general dynamic-shape compilation. Intermediate sequence
  lengths are intentionally unsupported.
* The ONNX dimension name is currently fixed to ``sequence_length``.
* The two specializations are compiled independently inside one program; this change shares
  initializers and runtime ownership, not generated kernels.
* Candidate selection is based on exact concrete input shapes. The first matching candidate wins.
* The fast evaluator path is an optimization only. Graphs that do not match its structural and
  capture-safety checks use the generic evaluator.
