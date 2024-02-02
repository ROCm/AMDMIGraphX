.. meta::
   :description: MIGraphX provides an optimized execution engine for deep learning neural networks
   :keywords: MIGraphX, ROCm, library, API, tool

.. _migraphx-driver:

=====================
MIGraphX driver
=====================

The MIGraphX driver is a command-line tool that allows you to utilize many of the MIGraphX core functions without having to write a program.
It can read, compile, run, and test the performance of a model with randomized data.

It is installed by default when you install MIGraphX. You can find it in `/opt/rocm/bin/migraphx-driver` or in 'AMDMIGraphX/build/bin/migraphx-driver' after building the source code.

.. _driver commands:

Commands
-----------

The table below summarizes the MIGraphX driver commands.

.. list-table:: commands
   .. header-rows: 1

   *  - Command
      - Description
   *  - op
      - Prints all operators of MIGraphX when followed by the option `--list` or `-l`
   *  - params
      - Prints the input and output parameter shapes
   *  - run
      - Compiles, allocates parameters, evaluates, and prints input graph
   *  - read
      - Loads and prints input graph
   *  - compile
      - Compiles and prints input graph
   *  - verify
      - Runs reference and GPU implementations and checks outputs for consistency
   *  - perf
      - Compiles and runs input graph followed by printing the performance report

Options
----------

The table below summarizes the various options to be used with the :ref:`MIGraphX driver commands <driver commands>`.
To learn which options can be used with which commands, see the :ref:`MIGraphX driver options <driver-options>`.

.. list-table:: commands
   .. header-rows: 1

   *  - Option
      - Description
   *  - --help | -h
      - Prints help section.
   *  - --model <resnet50|inceptionv3|alexnet>
      - Loads one of the three default models.
   *  - --onnx
      - Loads the file as an onnx graph.
   *  - --tf
      - Loads the file as a tensorflow graph.
   *  - --migraphx
      - Loads the file as a migraphx graph.
   *  - --migraphx-json
      - Loads the file as a migraphx JSON graph.
   *  - --batch
      - Sets batch size for a static model. Sets the batch size at runtime for a dynamic batch model.
   *  - --nhwc
      - Treats tensorflow format as nhwc.
   *  - --nchw
      - Treats tensorflow format as nchw.
   *  - --skip-unknown-operators	
      - Skips unknown operators when parsing and continues to parse.
   *  - --trim | -t
      - Trims instructions from the end.
   *  - --optimize | -O
      - Optimizes read
   *  - --graphviz | -g
      - Prints a graphviz representation
   *  - --brief
      - Makes the output brief
   *  - --cpp
      - Prints the program in .cpp format
   *  - --json
      - Prints the program in .json format
   *  - --text
      - Prints the program in .txt format
   *  - --binary
      - Prints the program in binary format
   *  - --output | -o
      - Writes output in a file
   *  - --fill0
      - Fills parameter with 0s
   *  - --fill1
      - Fills parameter with 1s



















