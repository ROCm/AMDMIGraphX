MIGraphX Driver
===============

read
----

.. program:: migraphx-driver read

Loads and prints input graph.

.. include:: ./driver/read.rst

compile
-------

.. program:: migraphx-driver compile

Compiles and prints input graph.

.. include:: ./driver/compile.rst

run
---

.. program:: migraphx-driver run

Loads and prints input graph.

.. include:: ./driver/compile.rst

perf
----

.. program:: migraphx-driver perf

Compiles and runs input graph then prints performance report.

.. include:: ./driver/compile.rst

.. option::  --iterations, -n [unsigned int]

Number of iterations to run for perf report (Default: 100)

verify
------

.. program:: migraphx-driver verify

Runs reference and CPU or GPU implementations and checks outputs for consistency.

.. include:: ./driver/compile.rst

.. option::  --rms-tol [double]

Tolerance for RMS error (Default: 0.001)

.. option::  --atol [double]

Tolerance for elementwise absolute difference (Default: 0.001)

.. option::  --rtol [double]

Tolerance for elementwise relative difference (Default: 0.001)

.. option::  -i, --per-instruction

Verify each instruction

.. option::  -r, --reduce

Reduce program and verify

roctx
----

.. program:: migraphx-driver roctx

Provides marker information for each operation, allowing MIGraphX to be used with `rocprof <https://rocmdocs.amd.com/en/latest/ROCm_Tools/ROCm-Tools.html>`_ for performance analysis.
This allows user to get GPU-level kernel timing information.
An example command line combined with rocprof for tracing purposes is given below:

.. code-block:: bash

    /opt/rocm/bin/rocprof --hip-trace --roctx-trace --flush-rate 1ms --timestamp on -d <OUTPUT_PATH> --obj-tracking on /opt/rocm/bin/migraphx-driver roctx <ONNX_FILE> <MIGRAPHX_OPTIONS>

After `rocprof` is run, the output directory will contain trace information for HIP, HCC and ROCTX in seperate `.txt` files.
To understand the interactions between API calls, it is recommended to utilize `roctx.py` helper script as desribed in :ref:`dev/tools:rocTX` section. 

.. include:: ./driver/compile.rst