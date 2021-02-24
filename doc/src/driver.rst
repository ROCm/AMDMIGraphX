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

.. option::  --tolerance [double]

Tolerance for errors (Default: 80)

.. option::  -i, --per-instruction

Verify each instruction

.. option::  -r, --reduce

Reduce program and verify
