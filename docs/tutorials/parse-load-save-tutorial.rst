.. meta::
  :description: Parse, load, and save MIGraphX programs
  :keywords: MIGraphX, tutorial, ONNX, parse, load, save, C++

********************************************************************
Parse, load, and save a model
********************************************************************

This tutorial walks through the
``examples/migraphx/cpp_parse_load_save`` example in the MIGraphX repository.
You learn how to parse an ONNX model into a MIGraphX program, inspect the
graph, save it to MessagePack format, and reload it later.

The example source is at
`examples/migraphx/cpp_parse_load_save <https://github.com/ROCm/AMDMIGraphX/tree/develop/examples/migraphx/cpp_parse_load_save>`__
in the MIGraphX GitHub repository.

What you will learn
====================================================================

* Parse an ONNX file into a ``migraphx::program``.
* Save a program to MessagePack (``.mxr``) format.
* Load a previously saved program from MessagePack or JSON format.
* Swap in your own ONNX model by changing the input file path.

Prerequisites
====================================================================

* MIGraphX installed. See :doc:`MIGraphX on ROCm installation <../install/install-migraphx>`.
* A C++ compiler and CMake 3.5 or later.
* An ONNX model file. The tutorial uses any compatible ONNX file you provide
  as ``input_model.onnx``.

Build the example
====================================================================

From the ``examples/migraphx/cpp_parse_load_save`` directory:

.. code-block:: shell

   mkdir build
   cd build
   cmake ..
   make

This produces an executable named ``parse_load_save``.

Parse an ONNX model
====================================================================

By default, the program parses the input file as ONNX and prints the internal
graph structure. Replace ``input_model.onnx`` with the path to your model:

.. code-block:: shell

   ./parse_load_save input_model.onnx --parse onnx

The C++ API call used in the example is:

.. code-block:: cpp

   migraphx::program p;
   migraphx::onnx_options options;
   p = parse_onnx(input_file, options);

Computation graphs saved in a compatible serialized format, such as ONNX, can
be read by MIGraphX to create a runnable program.

Save a program
====================================================================

Save the parsed program to MessagePack format. Replace ``output_name`` with
your preferred base filename (the ``.mxr`` extension is added automatically):

.. code-block:: shell

   ./parse_load_save input_model.onnx --parse onnx --save output_name

The program writes ``output_name.mxr`` in the current directory.

Load a saved program
====================================================================

Load a program that was previously saved in MessagePack format:

.. code-block:: shell

   ./parse_load_save output_name.mxr --load msgpack

To load a program saved in JSON format:

.. code-block:: shell

   ./parse_load_save saved_program.json --load json

Use your own model and data
====================================================================

To adapt this example for your workflow:

1. Replace ``input_model.onnx`` with the path to your ONNX model file.
2. If your model has dynamic or unspecified dimensions, configure
   ``migraphx::onnx_options`` before parsing. For example, set a default batch
   size with ``options.set_default_dim_value(batch)``, where ``batch`` is your
   batch size integer.
3. After parsing, compile the program for your target and run inference. See
   the C++ MNIST example at
   ``examples/vision/cpp_mnist/`` for compile and evaluate steps.

Command-line reference
====================================================================

The ``parse_load_save`` executable supports the following usage:

.. code-block:: text

   ./parse_load_save <input_file> [options]
   options:
           --parse onnx
           --load  json/msgpack
           --save  <output_file>

Next steps
====================================================================

* Complete an end-to-end inference workflow in the C++ MNIST example
  documented at ``examples/vision/cpp_mnist/``.
* Use :doc:`MIGraphX driver <../migraphx-driver>` to inspect and verify your
  model from the command line without writing C++ code.
* See :doc:`MIGraphX examples <./MIGraphX-examples>` for additional examples
  by domain.
