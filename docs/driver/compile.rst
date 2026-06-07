.. option::  --fill0 [std::vector<std::string>]

Fill parameter with 0s

.. option::  --fill1 [std::vector<std::string>]

Fill parameter with 1s

.. option::  --gpu

Compile on the gpu

.. option::  --cpu

Compile on the cpu

.. option::  --ref

Compile on the reference implementation

.. option::  --gpu-arch [std::string]

Cross-compile for the given GPU architecture (e.g. ``gfx942``) without requiring a physical device. Only applies to the ``gpu`` target.

.. option::  --gpu-num-cus [std::size_t] (Default: 120)

Number of compute units to assume for cross-compilation. Only used when ``--gpu-arch`` is set.

.. option::  --gpu-num-chiplets [std::size_t] (Default: 1)

Number of chiplets (XCCs) to assume for cross-compilation. Only used when ``--gpu-arch`` is set.

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
