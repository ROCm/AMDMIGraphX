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

.. option::  --gpu-arch-params [std::string]

Device properties to assume for cross-compilation, as a JSON object (e.g. ``"{arch:gfx942, num_cu:120, num_chiplets:1, max_threads_per_cu:2048, max_threads_per_block:1024}"``). Overrides ``--gpu-arch``, ``--gpu-num-cus`` and ``--gpu-num-chiplets`` for any keys present. Specifying ``arch`` here is sufficient to enable cross-compilation without ``--gpu-arch``.

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

.. option::  --encode-weights [std::string]

Encode the raw weight files in the given directory into the compiled program by replacing each ``external_weight`` op with a literal, producing a self-contained model. The input may be an ONNX model parsed with ``--weight-params`` or a previously-saved template ``.mxr``; the result is written to the path given by ``--output``. The program must contain ``external_weight`` ops (parse with ``--weight-params``) for there to be anything to encode.
