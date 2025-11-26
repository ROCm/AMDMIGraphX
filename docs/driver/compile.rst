.. option::  --fill0 [std::vector<std::string>]

Fill parameter with 0s

.. option::  --fill1 [std::vector<std::string>]

Fill parameter with 1s

.. option:: --load-arg [std::vector<std::string>]

Load arguments for the model (format: "@name filename")

.. option::  --gpu

Compile on the gpu

.. option::  --cpu

Compile on the cpu

.. option::  --ref

Compile on the reference implementation

.. option::  --enable-offload-copy

Enable implicit offload copying

.. option::  --disable-fast-math

Disable fast math optimization

.. option:: --exhaustive-tune

Perform an exhaustive search to find the fastest version of generated kernels for selected backend

.. option::  --fp16

Quantize for fp16

.. option::  --bf16

Quantize for bf16

.. option::  --int8

Quantize for int8

.. option:: --fp8

Quantize for Float8E4M3FNUZ type

.. option:: --int4-weights

Quantize weights for int4
