.. meta::
   :description: MIGraphX library precision support
   :keywords: MIGraphX, ROCm, API library, API reference, data type, support, precision

.. _migraphx-data-type-support:

******************************************
MIGraphX precision support
******************************************

This topic lists the data type support for the MIGraphX library on AMD GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _migraphx-input-output-type-support:

Supported data types
====================

The following table lists the primitive compute types supported by MIGraphX —
types that can be specified as tensor element types and computed on directly.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}

.. _migraphx-quantized-model-formats:

Supported quantized model formats
==================================

In addition to the compute types above, MIGraphX can execute models that use
the following quantized weight formats. These are not native compute types —
MIGraphX unpacks them internally around compute operations rather than
operating on them directly.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    *
      - Format
      - Description
    *
      - MXFP4 (E2M1)
      - OCP microscaling 4-bit floating-point format. Supported via the ONNX
        ``MXQuantizeDequantize`` and ``DynamicScale`` operators.
    *
      - INT4
      - 4-bit integer format used in bitsandbytes and GGML quantized models.
        Supported via the ONNX ``MatMulBnb4`` and ``MatMulNBits`` operators.
    *
      - NF4
      - NormalFloat 4-bit format used in bitsandbytes quantized models.
        Supported via the ONNX ``MatMulBnb4`` operator.
