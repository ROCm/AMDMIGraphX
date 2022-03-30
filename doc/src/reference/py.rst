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
------

.. py:class:: target()

    This represents the compilation target.

.. py:function:: get_target(name)

    Constructs the target.

    :param str name: The name of the target to construct. This can either be 'cpu' or 'gpu'.

    :rtype: target


module
------
.. py:method:: print()

    This prints the contents of the module as list of instructions

.. py:method:: add_instruction(op, args, mod_args=[])
    
    Adds instruction into the module

    :param operation op: 'migraphx::op' to be added as instruction
    :param list[instruction] args: list of inputs to the op 
    :param list[module] mod_args: optional list of module arguments to the operator
    :rtype instruction

.. py:method:: add_parameter(name, shape)
    
    Adds a parameter to the module with provided name and shape

    :param str name: name of the parameter
    :param shape shape: shape of the parameter
    :rtype instruction

.. py:method:: add_return(args)

    Adds a return instruction into the module

    :param list[instruction] args: instruction arguments which need to be returned from the module
    :rtype instruction


program
-------

.. py:class:: program()

    Represents the computation graph to be compiled and run.

.. py:method:: clone()

    Make a copy of the program

    :rtype: program

.. py:method:: get_parameter_names()
 
    Get all the input arguments' or parameters' names to the program as a list

    :rtype list[str]

.. py:method:: get_parameter_shapes()

    Get the shapes of all the input parameters in the program.

    :rtype: dict[str, shape]

.. py:method:: get_output_shapes()

    Get the shapes of the final outputs of the program.

    :rtype: list[shape]

.. py:method:: compile(t, offload_copy=True, fast_math=True)

    Compiles the program for the target and optimizes it.

    :param target t: This is the target to compile the program for.
    :param bool offload_copy: For targets with offloaded memory(such as the gpu), this will insert instructions during compilation to copy the input parameters to the offloaded memory and to copy the final result from the offloaded memory back to main memory.
    :param bool fast_math: Optimize math functions to use faster approximate versions. There may be slight accuracy degredation when enabled.

.. py:method:: get_main_module()
    
    Get main module of the program

    :rtype module

.. py:method:: create_module(name)
    
    Create and add a module of provided name into the program

    :param str name : name of the new module  
    :rtype module

.. py:method:: run(params)

    Run the program.

    :param params: This is a map of the input parameters which will be used when running the program.
    :type params: dict[str, argument]

    :return: The result of the last instruction.
    :rtype: list[argument]

.. py:method:: sort()

    Sort the modules of the program such that instructions appear in correct occurrence.

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


op
--
.. py::class:: op(name, kwargs)

    Construct an operation with name and arguments
    
    :param str name : name of the operation, must be supported by MIGraphX
    :param dict[str, any] kwargs: arguments to the operation  
    :rtype operation



parse_onnx
----------

.. py:function:: parse_onnx(filename, default_dim_value=1, map_input_dims={}, skip_unknown_operators=false, print_program_on_error=false, max_loop_iterations=10)

    Load and parse an onnx file.

    :param str filename: Path to file.
    :param str default_dim_value: default batch size to use (if not specified in onnx file).
    :param str map_input_dims: Explicitly specify the dims of an input.
    :param str skip_unknown_operators: Continue parsing onnx file if an unknown operator is found.
    :param str print_program_on_error: Print program if an error occurs.
    :param int max_loop_iterations: Maximum iteration number for the loop operator
    :rtype: program

parse_tf
--------

.. py:function:: parse_tf(filename, is_nhwc=True, batch_size=1, map_input_dims=dict(), output_names=[])

    Load and parse an tensorflow protobuf file file.

    :param str filename: Path to file.
    :param bool is_nhwc: Use nhwc as default format.
    :param str batch_size: default batch size to use (if not specified in protobuf).
    :param dict[str, list[int]] map_input_dims: Optional arg to explictly specify dimensions of the inputs
    :param list[str] output_names:  Optional argument specify names of the output nodes
    :rtype: program

load
----

.. py:function:: load(filename, format='msgpack')

    Load a MIGraphX program

    :param str filename: Path to file.
    :param str format: Format of file. Valid options are msgpack or json.

    :rtype: program

save
----

.. py:function:: save(p, filename, format='msgpack')

    Save a MIGraphX program

    :param program p: Program to save.
    :param str filename: Path to file.
    :param str format: Format of file. Valid options are msgpack or json.

