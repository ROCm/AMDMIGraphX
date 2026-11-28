.. meta::
   :description: MIGraphX driver options
   :keywords: MIGraphX, ROCm, driver, options

.. _driver-options:

Driver options
===============

This document lists the MIGraphX driver commands along with the eligible options.

read
----

.. program:: migraphx-driver read

Loads and prints input graph.

.. include:: ../driver/read.rst

compile
-------

.. program:: migraphx-driver compile

Compiles and prints input graph.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

recompile
---------

.. program:: migraphx-driver recompile

Recompiles the GPU kernels of a compiled program, which was compiled with ``--recompilable``, for the same GPU or
another one, and writes the result to the output file. The kernels of the input file are not loaded, so the file can be
for a GPU that is not in the machine. The output file, ``--output`` or ``-o``, is required. See :ref:`recompile-mxr`.

.. include:: ../driver/read.rst

.. option::  --gpu-arch [std::string]

Recompile for the given GPU architecture (e.g. ``gfx942``) without requiring a physical device. Without it, the kernels are recompiled for the GPU in the machine.

.. option::  --gpu-num-cus [std::size_t] (Default: 120)

Number of compute units to assume for the architecture. Only used when ``--gpu-arch`` is set.

.. option::  --gpu-num-chiplets [std::size_t] (Default: 1)

Number of chiplets (XCCs) to assume for the architecture. Only used when ``--gpu-arch`` is set.

.. option::  --gpu-wavefront-size [std::size_t] (Default: 0)

Wavefront size to assume for the architecture (32 or 64; 0 = infer from the architecture). Only used when ``--gpu-arch`` is set.

.. option::  --gpu-arch-params [std::string]

Device properties to assume for the architecture, as a JSON object (same format as for ``compile``). Overrides ``--gpu-arch``, ``--gpu-num-cus``, ``--gpu-num-chiplets`` and ``--gpu-wavefront-size`` for any keys present.

.. option::  --exhaustive-tune

Recompile with exhaustive tuning. The options recorded in the program are not used.

.. option::  --backend-options [std::string]

Target-specific compile options, as a JSON object (e.g. ``"{problem_cache_files:[cache.json]}"``).

run
---

.. program:: migraphx-driver run

Loads and prints input graph.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

perf
----

.. program:: migraphx-driver perf

Compiles and runs input graph then prints performance report.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

.. option::  --iterations, -n [unsigned int]

Sets number of iterations to run for perf report (Default: 100)

time
----

.. program:: migraphx-driver time

Compiles and runs input graph then prints the total execution time.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

.. option::  --iterations, -n [unsigned int]

Sets number of iterations to run (Default: 100)

.. option::  --buffers, -b [unsigned int]

Sets number of parameter buffer sets to rotate through between iterations to avoid cache reuse (Default: 1)

verify
------

.. program:: migraphx-driver verify

Runs reference and CPU or GPU implementations and checks outputs for consistency.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

.. option::  --rms-tol [double]

Sets tolerance for RMS error (Default: 0.001)

.. option::  --atol [double]

Sets tolerance for elementwise absolute difference (Default: 0.001)

.. option::  --rtol [double]

Sets tolerance for elementwise relative difference (Default: 0.001)

.. option::  -i, --per-instruction

Verifies each instruction

.. option::  -r, --reduce

Reduces program and verifies

.. option::  -b, --bisect

Uses binary search to locate the first failing step.

.. option::  -l, --layerwise

Compares corresponding reference and target layers in one execution.

.. option:: --ref-use-double

Converts floating point values to double for the ref target

.. _roctx:

roctx
------

.. program:: migraphx-driver roctx

``roctx`` provides marker information for each operation which allows MIGraphX to be used with :doc:`rocprof <rocprofiler:rocprofv1>` for performance analysis.
This allows you to get GPU-level kernel timing information.
Here is how you can use ``roctx`` combined with :doc:`rocprof <rocprofiler:rocprofv1>` for tracing:

.. code-block:: bash

    /opt/rocm/bin/rocprof --hip-trace --roctx-trace --flush-rate 1ms --timestamp on -d <OUTPUT_PATH> --obj-tracking on /opt/rocm/bin/migraphx-driver roctx <ONNX_FILE> <MIGRAPHX_OPTIONS>

Running :doc:`rocprof <rocprofiler:rocprofv1>` generates trace information for HIP, HCC and ROCTX in separate ``.txt`` files.
To understand the interactions between API calls, utilize the :ref:`roctx.py <tools>` helper script.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst
