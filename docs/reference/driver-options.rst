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

.. option:: --detailed, -d

Show a more detailed summary report

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

.. option:: -b, --bisect

Bisects program and verifies

.. option:: --ref-use-double

Converts floating point values to double for the ref target

.. option:: --compiled-model, -c [std::string]

Compiled model to use

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

time
----

.. program:: migraphx-driver time

Compiles, allocates parameters, runs model, and prints total execution time.

.. include:: ../driver/read.rst
.. include:: ../driver/compile.rst

.. option::  --iterations, -n [unsigned int]

Number of iterations to run (Default: 100)

params
------

.. program:: migraphx-driver params

Prints the input and output parameter shapes.

.. include:: ../driver/read.rst

op
--

.. program:: migraphx-driver op

Displays information about MIGraphX operators.

.. option:: <MIGraphX operator name>

Name of the operator to display

.. option:: --list, -l

List all the operators of MIGraphX

onnx
----

.. program:: migraphx-driver onnx

Lists ONNX operators supported by MIGraphX.

.. option:: --list, -l

List all onnx operators supported by MIGraphX

tf
--

.. program:: migraphx-driver tf

Lists TensorFlow operators supported by MIGraphX.

.. option:: --list, -l

List all tf operators supported by MIGraphX
