
C++ Reference
=============

shape
-----

.. doxygenenum:: migraphx_shape_datatype_t

.. doxygenstruct:: migraphx::shape

argument
--------

.. doxygenstruct:: migraphx::argument

target
------

.. doxygenstruct:: migraphx::target

program
-------

.. doxygenstruct:: migraphx::program_parameter_shapes

.. doxygenstruct:: migraphx::program_parameters

.. doxygenstruct:: migraphx_compile_options

.. doxygenstruct:: migraphx::program

quantize
--------

.. doxygenstruct:: migraphx::quantize_op_names

.. doxygenfunction:: migraphx::quantize_fp16(const program&)

.. doxygenfunction:: migraphx::quantize_fp16(const program&, const quantize_op_names&)

.. doxygenstruct:: migraphx::quantize_int8_options

.. doxygenfunction:: migraphx::quantize_int8

parse_onnx
----------

.. doxygenstruct:: migraphx::onnx_options

.. doxygenfunction:: migraphx::parse_onnx(const char *)

.. doxygenfunction:: migraphx::parse_onnx(const char *, const migraphx::onnx_options&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const std::string&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const std::string&, const migraphx::onnx_options&)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const void *, size_t)

.. doxygenfunction:: migraphx::parse_onnx_buffer(const void *, size_t, const migraphx::onnx_options&)

load
----

.. doxygenstruct:: migraphx_file_options

.. doxygenfunction:: migraphx::load(const char *)

.. doxygenfunction:: migraphx::load(const char *, migraphx_file_options)

save
----

.. doxygenfunction:: migraphx::save(const program&, const char *)

.. doxygenfunction:: migraphx::save(const program&, const char *, migraphx_file_options)

