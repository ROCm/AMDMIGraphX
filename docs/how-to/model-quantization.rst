.. meta::
  :description: Quantize models with MIGraphX to FP16, BF16, INT8 or INT4 weights
  :keywords: MIGraphX, quantization, fp16, bf16, int8, int4, calibration, ROCm

********************************************************************
Quantize models
********************************************************************

MIGraphX can convert an FP32 (or FP16) ONNX model to a lower precision while it compiles it. Lower precision makes
inference faster and the model smaller, at some cost in accuracy. This guide explains each built-in option, how to
use it from the command line, Python and C++, and how to check that the result is still accurate.

In the examples, ``data`` and ``1 3 224 224`` are an example input name and shape. Use the input name and shape of
your own model.

Choose a precision
====================================================================

.. list-table::
   :header-rows: 1

   * - Option
     - Driver flag
     - Python
     - C++
     - Needs calibration data
     - Typical use
   * - **FP16**
     - ``--fp16``
     - ``quantize_fp16``
     - ``quantize_fp16``
     - No
     - **Recommended first choice.** Usually no measurable accuracy loss.
   * - **BF16**
     - ``--bf16``
     - ``quantize_bf16``
     - ``quantize_bf16``
     - No
     - Alternative to FP16 with a wider value range.
   * - **INT8**
     - ``--int8``
     - ``quantize_int8``
     - ``quantize_int8``
     - **Yes**
     - Can give the highest speed for convolution networks. Validate accuracy per model.
   * - **INT4 weights**
     - ``--int4-weights``
     - Not available
     - Not available
     - No
     - Experimental. See :ref:`quantization-int4-weights`.

The recommended approach:

#. Start with **FP16**. Check accuracy on your own data (:ref:`quantization-verify`).
#. If you need more speed, try **INT8** with a few real, representative inputs as calibration data. Compare accuracy
   with the FP32 model before deploying.

How built-in quantization works
====================================================================

* Quantization is applied to the **uncompiled** program, before ``compile``. A program that is already compiled (a
  saved ``.mxr`` file) cannot be quantized again.
* MIGraphX inserts the conversions, then the compiler fuses them into the GPU kernels. Check the compiled program to
  confirm which kernels run in the new precision (:ref:`quantization-check-kernels`).
* **Start from an FP32 or FP16 model.** Do not apply these options to a model that is already quantized (one that
  contains ``QuantizeLinear``/``DequantizeLinear``, ``QLinearConv`` or other quantized operators): run such models as
  shipped. See :ref:`quantization-troubleshooting`.
* **INT8 needs fixed input shapes.** Set them with ``--input-dim`` (driver), ``map_input_dims`` (Python) or
  ``set_input_parameter_shape`` (C++).

What built-in quantization does not support
--------------------------------------------------------------------

**INT8:**

* Only symmetric quantization (zero point 0). There is no asymmetric option.
* Only one scale per tensor. There are no per-channel scales, including for weights.
* Scales come only from the largest absolute value. There is no percentile or entropy calibration.
* INT8 converts only convolutions and matrix multiplies, and you cannot exclude individual layers (for example
  attention) from it.

**INT4 weights:**

* Only one scale per weight tensor. There are no per-channel or per-group scales.
* Only round-to-nearest. There are no error-compensating algorithms such as GPTQ or AWQ.
* Command line only. There is no Python or C++ API.
* Weights are only stored in 4 bits. Activations stay in FP32/FP16, and the weights are expanded before each matrix
  multiply.

FP16 and BF16
====================================================================

All floating-point operations are converted to FP16 (or BF16). No calibration data is needed.

**Command line**

.. code-block:: bash

   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 --fp16 -o model_fp16.mxr

**Python**

.. code-block:: python

   import migraphx

   prog = migraphx.parse_onnx("model.onnx", map_input_dims={"data": [1, 3, 224, 224]})
   migraphx.quantize_fp16(prog)          # or migraphx.quantize_bf16(prog)
   prog.compile(migraphx.get_target("gpu"))

**C++**

.. code-block:: cpp

   migraphx::program prog = migraphx::parse_onnx("model.onnx");
   migraphx::quantize_fp16(prog);        // or migraphx::quantize_bf16(prog)
   prog.compile(migraphx::target("gpu"));

Converting only some operations
--------------------------------------------------------------------

To keep part of the model in FP32, list the operation types to convert. The list uses **MIGraphX operation names, not
ONNX names**: MIGraphX translates each ONNX operator into its own operations, and the list is matched against those.

.. list-table::
   :header-rows: 1

   * - ONNX operator
     - MIGraphX operation name
   * - ``Conv``
     - ``convolution``
   * - ``MatMul``, ``Gemm``
     - ``dot``
   * - ``ConvTranspose``
     - ``convolution_backwards``

Some ONNX operators are split into several basic operations. For example, ``BatchNormalization`` becomes ``add``,
``mul``, ``sub`` and ``rsqrt``. To see the names used in your model, parse it and count the operations:

.. code-block:: bash

   migraphx-driver read model.onnx --input-dim @data 1 3 224 224 | grep -oE '= [a-z_0-9]+' | sort | uniq -c | sort -rn

.. code-block:: python

   migraphx.quantize_fp16(prog, ins_names=["convolution", "dot"])

.. code-block:: cpp

   migraphx::quantize_op_names names;
   names.add("convolution");
   names.add("dot");
   migraphx::quantize_fp16(prog, names);

**FP16 or BF16?** FP16 keeps more precision. BF16 keeps the FP32 value range. Use BF16 if a model produces ``inf`` or
``NaN`` with FP16.

INT8
====================================================================

What gets quantized
--------------------------------------------------------------------

* The inputs of **convolution** and **matrix-multiply** (``MatMul``, ``Gemm``) operations are converted to INT8. All
  other operations stay in their original precision.
* Weights and activations use **one scale per tensor** and a zero point of 0 (symmetric).
* Each scale comes from the **largest absolute value** seen while running the calibration data: that value maps to
  127, so the quantization step is ``max(|x|) / 127``.

Calibration data: the most important step
--------------------------------------------------------------------

INT8 accuracy depends on the calibration data.

* **Use real inputs**, preprocessed exactly as at inference time.
* **A few representative samples are usually enough.** More samples do not always help: a single extreme value sets
  the scale of a whole tensor, so unusual inputs can reduce accuracy.
* **Always provide calibration data.** Without it, MIGraphX falls back to a default scale and accuracy can collapse.
* On the command line, ``--int8`` calibrates with **synthetic data** unless you pass a real input with ``--load-arg``
  (below). Synthetic data does not reflect the value ranges of real inputs, so the scales can be poor and accuracy can
  drop a lot.

Command line
--------------------------------------------------------------------

#. Save one representative input with Python (repeat for each model input):

   .. code-block:: python

      import numpy as np
      import migraphx

      image = np.load("preprocessed_image.npy").astype(np.float32)    # shape (1, 3, 224, 224)
      migraphx.argument.save(migraphx.argument(image), "calib_input.mxa")

#. Compile with ``--int8``, passing that input:

   .. code-block:: bash

      migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 \
          --int8 --load-arg @data calib_input.mxa -o model_int8.mxr

The command line uses one calibration input. To calibrate with several samples, use the Python or C++ API.

Python
--------------------------------------------------------------------

.. code-block:: python

   import numpy as np
   import migraphx

   prog = migraphx.parse_onnx("model.onnx", map_input_dims={"data": [1, 3, 224, 224]})
   gpu = migraphx.get_target("gpu")

   # A few representative, preprocessed inputs, each with the model's input shape
   calibration = [{"data": migraphx.argument(img)} for img in calibration_images]

   migraphx.quantize_int8(prog, gpu, calibration)    # before compile
   prog.compile(gpu)
   migraphx.save(prog, "model_int8.mxr")             # load later with migraphx.load()

   result = prog.run({"data": migraphx.argument(input_image)})
   logits = np.array(result[0])

C++
--------------------------------------------------------------------

.. code-block:: cpp

   #include <migraphx/migraphx.hpp>

   migraphx::onnx_options onnx_opts;
   onnx_opts.set_input_parameter_shape("data", {1, 3, 224, 224});
   migraphx::program prog = migraphx::parse_onnx("model.onnx", onnx_opts);
   migraphx::target gpu("gpu");
   auto shapes = prog.get_parameter_shapes();

   migraphx::quantize_int8_options quant_opts;
   for(auto& sample : calibration_samples)           // std::vector<std::vector<float>>
   {
       migraphx::program_parameters calib;
       calib.add("data", migraphx::argument(shapes["data"], sample.data()));
       quant_opts.add_calibration_data(calib);       // once per calibration sample
   }
   migraphx::quantize_int8(prog, gpu, quant_opts);   // before compile

   migraphx::compile_options comp_opts;
   comp_opts.set_offload_copy();
   prog.compile(gpu, comp_opts);

INT8 together with FP16
--------------------------------------------------------------------

``--fp16 --int8`` (or ``quantize_fp16`` followed by ``quantize_int8``) runs convolutions and matrix multiplies in
INT8 and the remaining operations in FP16. Use FP16, not BF16, for this combination: with ``--bf16 --int8`` the INT8
step is not applied.

What to expect
--------------------------------------------------------------------

* **Convolution networks** (for example ResNet) usually work well with INT8 and are where it helps most.
* **Networks with depthwise convolutions** (for example MobileNet) and **transformers** (for example BERT) can lose
  noticeable accuracy with INT8, even with good calibration data. Measure on your data, and use FP16 if the loss is
  too high.

.. _quantization-int4-weights:

INT4 weights (experimental)
====================================================================

``--int4-weights`` (command line only) stores the constant weights of matrix multiplies and convolutions as 4-bit
integers. Activations are not quantized, and no calibration data is needed.

How the weights are quantized
--------------------------------------------------------------------

Each weight tensor gets one scale and one zero point (asymmetric min-max quantization to the unsigned range 0-15):

.. code-block:: text

   low   = min(smallest weight, 0)
   high  = max(largest weight, 0)
   scale = (high - low) / 15
   zero_point = round(-low / scale)
   q = round(w / scale) + zero_point          # stored as 4 bits, two values per byte

At run time the weights are converted back (``(q - zero_point) * scale``) before the matrix multiply or convolution.
The range always includes 0, so a weight of 0 is stored exactly.

A weight is converted only if it is a constant FP32 or FP16 input of a matrix multiply or convolution, it is not
broadcast, and its last dimension has an even size.

.. warning::

   Use with care.

   * Many weights are left unconverted, without a message: convolution weights whose last dimension (kernel width) is
     odd, and MatMul weights that are broadcast across a batch dimension. This is common in transformer models. As a
     result, many models see no change in size or speed.
   * Where weights are converted, one scale per tensor is coarse and can reduce accuracy severely.
   * Always check accuracy before using it.

.. code-block:: bash

   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 --int4-weights -o model_int4w.mxr

.. _quantization-verify:

Verify a quantized model
====================================================================

Measure accuracy on your own data (recommended)
--------------------------------------------------------------------

Run the quantized model and the FP32 model on a set of real inputs and compare the metric you care about (top-1
accuracy, F1, and so on). This is the only reliable test. Synthetic inputs can hide large accuracy losses.

Quick numerical check with the driver
--------------------------------------------------------------------

Compile the quantized model first, then compare it with the original model using ``verify -c``:

.. code-block:: bash

   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 \
       --int8 --load-arg @data calib_input.mxa -o model_int8.mxr
   migraphx-driver verify model.onnx --gpu --input-dim @data 1 3 224 224 -c model_int8.mxr --rms-tol 0.1

* ``-c`` checks the compiled, quantized program against an FP32 reference run of ``model.onnx``.
* **Set** ``--rms-tol`` **yourself.** Without it, ``verify -c`` uses the FP32 tolerance (0.001), which quantized
  models normally exceed.
* **Do not use** ``verify --int8`` **or** ``verify --int4-weights``. These options do not quantize the program being
  checked, so they report a pass even when the quantized model is inaccurate. (``verify --fp16`` and
  ``verify --bf16`` do quantize.)
* **Read the output, not the exit status.** Look for ``MIGraphX verification passed successfully`` or ``FAILED`` with
  the ``RMS Error``. The driver can return exit status 0 when verification fails.
* ``verify`` uses synthetic inputs. Treat it as a sanity check, not an accuracy measurement.

.. _quantization-check-kernels:

Check which kernels run in the new precision
--------------------------------------------------------------------

.. code-block:: bash

   migraphx-driver read model_int8.mxr | grep -c quant_      # INT8 convolution / matrix-multiply kernels
   migraphx-driver read model_fp16.mxr | grep -c half_type   # FP16 tensors

A count of 0 means the conversion did not take effect.

Measure speed
--------------------------------------------------------------------

.. code-block:: bash

   migraphx-driver perf model_int8.mxr --gpu -n 100

Deploy the file you validated
--------------------------------------------------------------------

Save the compiled program (``-o`` on the command line, ``migraphx.save`` in Python) and deploy that exact file. INT8
programs compiled separately from the same model can give slightly different results.

.. _quantization-troubleshooting:

Troubleshoot
====================================================================

.. list-table::
   :header-rows: 1

   * - Symptom
     - Cause
     - What to do
   * - Large accuracy loss with ``--int8``
     - The driver calibrated with synthetic data
     - Pass a real input with ``--load-arg``, or calibrate through the API
   * - INT8 accuracy gets worse with more calibration samples
     - One extreme value sets the scale of a whole tensor
     - Use a few typical samples. Leave out unusual inputs
   * - ``verify --int8`` always passes
     - It does not quantize the program it checks
     - Compile with ``-o``, then ``verify -c <file.mxr> --rms-tol <tol>``
   * - ``verify`` prints ``FAILED`` but returns exit status 0
     - Exit status does not reflect the result
     - Check the output text
   * - No INT8 kernels after ``--bf16 --int8``
     - INT8 is not applied on top of BF16
     - Use ``--fp16 --int8``
   * - Output is all ``NaN`` after ``--fp16``
     - The model was already quantized
     - Run quantized models as shipped
   * - Worse accuracy after ``--int8`` on an INT8 model
     - The model was quantized twice
     - Run quantized models as shipped
   * - ``lens() called on a dynamic shape`` from INT8
     - Input shapes are not fixed
     - Set fixed shapes (``--input-dim``, ``map_input_dims``, ``set_input_parameter_shape``)
   * - ``QUANTIZE_INT8: only support DOT and CONVOLUTION operation``
     - Other operation names were passed to ``quantize_int8``
     - Omit the operation list (the default covers both)
   * - ``Quantization options are ignored as the program is already compiled``
     - The input is a compiled ``.mxr``. No output file is written
     - Quantize the ONNX model
   * - ``--int4-weights`` changes nothing
     - Weights were not eligible (see :ref:`quantization-int4-weights`)
     - Expected for many models
   * - Small differences between two INT8 builds
     - Separate compilations can differ slightly
     - Validate and deploy one saved ``.mxr``

Quick reference
====================================================================

.. code-block:: bash

   # FP16 / BF16
   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 --fp16 -o model_fp16.mxr

   # INT8 with a real calibration input (saved with migraphx.argument.save)
   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 \
       --int8 --load-arg @data calib_input.mxa -o model_int8.mxr

   # INT8 convolutions/matrix multiplies, FP16 elsewhere
   migraphx-driver compile model.onnx --gpu --input-dim @data 1 3 224 224 \
       --fp16 --int8 --load-arg @data calib_input.mxa -o model_fp16_int8.mxr

   # Check, inspect, time
   migraphx-driver verify model.onnx --gpu --input-dim @data 1 3 224 224 -c model_int8.mxr --rms-tol 0.1
   migraphx-driver read model_int8.mxr | grep -c quant_
   migraphx-driver perf model_int8.mxr --gpu -n 100

.. list-table::
   :header-rows: 1

   * - API
     - Python
     - C++
   * - FP16
     - ``migraphx.quantize_fp16(prog, ins_names=["all"])``
     - ``migraphx::quantize_fp16(prog[, quantize_op_names])``
   * - BF16
     - ``migraphx.quantize_bf16(prog, ins_names=["all"])``
     - ``migraphx::quantize_bf16(prog[, quantize_op_names])``
   * - INT8
     - ``migraphx.quantize_int8(prog, target, calibration)``
     - ``migraphx::quantize_int8(prog, target, quantize_int8_options)``

``calibration`` is a list of input dictionaries, for example ``[{"data": migraphx.argument(array)}, ...]``.
