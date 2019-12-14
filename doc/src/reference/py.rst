.. py:module:: migraphx

Python Reference
================

shape
-----

.. py:class:: shape(type, lens, strides=None)

    Describes the shape of a tensor. This includes size, layout, and data type/

.. py:method:: type()

    An integer that represents the type

    :rtype: int

.. py:method:: lens()

    A list of the lengths of the shape

    :rtype: list[int]

.. py:method:: strides()

    A list of the strides of the shape

    :rtype: list[int]

.. py:method:: elements()

    The number of elements in the shape

    :rtype: int

.. py:method:: bytes()

    The number of bytes the shape uses

    :rtype: int

.. py:method:: type_size()

    The number of bytes one element uses

    :rtype: int

.. py:method:: packed()

    Returns true if the shape is packed.

    :rtype: bool

.. py:method:: transposed()

    Returns true if the shape is transposed.

    :rtype: bool

.. py:method:: broadcasted()

    Returns true if the shape is broadcasted.

    :rtype: bool

.. py:method:: standard()

    Returns true if the shape is a standard shape. That is, the shape is both packed and not transposed.

    :rtype: bool

.. py:method:: scalar()

    Returns true if all strides are equal to 0 (scalar tensor).

    :rtype: bool


argument
--------

.. py:class:: argument(data)

    Construct an argument from a python buffer. This can include numpy arrays.

.. py:method:: get_shape()

    Returns the shape of the argument.

    :rtype: shape

.. py:method:: tolist()

    Convert the elements of the argument to a python list.

    :rtype: list


.. py:function:: generate_argument(s, seed=0)

    Generate an argument with random data.

    :param shape s: Shape of argument to generate.
    :param int seed: The seed used for random number generation

    :rtype: argument





target
--------

.. py:class:: target()

    This represents the compiliation target.

.. py:function:: get_target(name)

    Constructs the target.

    :param str name: The name of the target to construct. This can either be 'cpu' or 'gpu'.

    :rtype: target


program
-------

.. py:class:: program()

    Represents the computation graph to compiled and run.

.. py:method:: clone()

    Make a copy of the program

    :rtype: program

.. py:method:: get_parameter_shapes()

    Get the shapes of all the input parameters in the program.

    :rtype: dict[str, shape]

.. py:method:: get_shape()

    Get the shape of the final output of the program.

    :rtype: shape

.. py:method:: compile(t, offload_copy=True)

    Compiles the program for the target and optimizes it.

    :param target t: This is the target to compile the program for.
    :param bool offload_copy: For targets with offloaded memory(such as the gpu), this will insert instructions during compilation to copy the input parameters to the offloaded memory and to copy the final result from the offloaded memory back to main memory.

.. py:method:: run(params)

    Run the program.

    :param params: This is a map of the input parameters which will be used when running the program.
    :type params: dict[str, argument]

    :return: The result of the last instruction.
    :rtype: argument

.. py:function:: quantize_fp16(prog, ins_names=["all"])

    Quantize the program to use fp16.

    :param program prog: Program to quantize.
    :param ins_names: List of instructions to quantize.
    :type ins_names: list[str]


.. py:function:: quantize_int8(prog, t, calibration=[], ins_names=["dot", "convolution"])

    Quantize the program to use int8.

    :param program prog: Program to quantize.
    :param target t: Target that will be used to run the calibration data.
    :param calibration: Calibration data used to decide the parameters to the int8 optimization.
    :type calibration: list[dict[str, argument]]
    :param ins_names: List of instructions to quantize.
    :type ins_names: list[str]


parse_onnx
----------

.. py:function:: parse_onnx(filename)

    Load and parse an onnx file.

    :param str filename: Path to file.

    :rtype: program

parse_tf
----------

.. py:function:: parse_tf(filename, is_nhwc=True)

    Load and parse an tensorflow protobuf file file.

    :param str filename: Path to file.
    :param bool is_nhwc: Use nhwc as default format.

    :rtype: program

