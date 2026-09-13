.. meta::
  :description: Validate MIGraphX model outputs
  :keywords: MIGraphX, validation, verify, environment variables, ROCm

********************************************************************
Validate model outputs
********************************************************************

Use MIGraphX validation tools to compare GPU or CPU target output against the
reference implementation. Validation helps confirm that compilation and
quantization changes produce numerically consistent results.

Validate with migraphx-driver
====================================================================

The ``verify`` command compiles your model for the reference and target
backends, runs both, and checks that outputs match within configured
tolerances. See :doc:`MIGraphX driver <../migraphx-driver>` for full command
documentation.

Basic verification
--------------------------------------------------------------------

Verify an ONNX model against the default target:

.. code-block:: shell

   /opt/rocm/bin/migraphx-driver verify model.onnx --onnx

Replace ``model.onnx`` with your model path.

Verify on GPU with tolerance settings
--------------------------------------------------------------------

Set absolute tolerance (``atol``), relative tolerance (``rtol``), and
root-mean-square tolerance (``rms-tol``) when comparing outputs:

.. code-block:: shell

   /opt/rocm/bin/migraphx-driver verify model.onnx --onnx --gpu \
       --atol 1e-5 --rtol 1e-5 --rms-tol 0.001

Replace ``model.onnx`` with your model path. Default tolerance values are
``0.001`` for ``atol``, ``rtol``, and ``rms-tol``.

Verify after quantization
--------------------------------------------------------------------

Check that fp16 quantization preserves accuracy within your tolerances:

.. code-block:: shell

   /opt/rocm/bin/migraphx-driver verify model.onnx --onnx --gpu --fp16 \
       --atol 1e-3 --rtol 1e-3

Replace ``model.onnx`` with your model path.

Additional verify options
--------------------------------------------------------------------

The driver supports these validation-related options:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``--per-instruction`` or ``-i``
     - Verifies each instruction individually.
   * - ``--reduce`` or ``-r``
     - Reduces the program and verifies the reduced graph.
   * - ``--atol``
     - Sets tolerance for elementwise absolute difference (default: ``0.001``).
   * - ``--rtol``
     - Sets tolerance for elementwise relative difference (default: ``0.001``).
   * - ``--rms-tol``
     - Sets tolerance for root-mean-square error (default: ``0.001``).

Validation environment variables
====================================================================

Set these environment variables before running ``migraphx-driver verify`` or
your application to adjust validation behavior.

Output comparison
--------------------------------------------------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Environment variable
     - Values
   * - ``MIGRAPHX_VERIFY_ENABLE_ALLCLOSE``
     - ``1`` verifies range tolerance using ``allclose``. ``0`` returns to default behavior.
   * - ``MIGRAPHX_VERIFY_DUMP_DIFF``
     - ``1`` writes test output and reference output when they differ. ``0`` returns to default behavior.

Example:

.. code-block:: shell

   export MIGRAPHX_VERIFY_DUMP_DIFF=1
   /opt/rocm/bin/migraphx-driver verify model.onnx --onnx --gpu

Replace ``model.onnx`` with your model path.

Testing and debugging
--------------------------------------------------------------------

These variables apply when running MIGraphX verify tests or debugging
validation failures during development:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Environment variable
     - Values
   * - ``MIGRAPHX_TRACE_TEST``
     - ``1`` prints reference and target programs even when verify tests pass.
   * - ``MIGRAPHX_TRACE_TEST_COMPILE``
     - ``cpu`` or ``gpu`` turns on compile tracing for verify tests on the given target. Cannot be used with ``MIGRAPHX_TRACE_COMPILE``.
   * - ``MIGRAPHX_DUMP_TEST``
     - ``1`` writes the model under verification to an MXR file.

Example:

.. code-block:: shell

   export MIGRAPHX_VERIFY_DUMP_DIFF=1
   export MIGRAPHX_TRACE_TEST=1
   /opt/rocm/bin/migraphx-driver verify model.onnx --onnx --gpu

Replace ``model.onnx`` with your model path.

Graph validation during development
--------------------------------------------------------------------

When developing passes or matchers, enable module validation after pattern
matches:

.. code-block:: shell

   export MIGRAPHX_VALIDATE_MATCHES=1

See :doc:`MIGraphX environment variables <../reference/MIGraphX-dev-env-vars>`
for the complete list of validation, testing, and tracing variables.

See also
====================================================================

* :doc:`MIGraphX driver <../migraphx-driver>` for the ``verify`` command and
  tolerance options.
* :doc:`Precision support <../reference/MIGraphX-data-type-support>` for
  supported data types when validating quantized models.
* :doc:`Tune model performance <./model-performance-tuning>` when adjusting
  compilation options that can affect numerical output.
