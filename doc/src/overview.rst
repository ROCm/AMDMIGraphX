Overview
========

MIGraphX provides an optimized execution engine for deep learning neural networks.

Building a program
------------------

A program consists of a set of instructions to be executed when calling `eval <migraphx::program::eval>`. Each instruction has an associated `operation <migraphx::operation>` which represents the computation to be performed by the instruction.

We can start by building a simple program to add two numbers together::

    program p;
    instruction_ref one = p.add_literal(1);
    instruction_ref two = p.add_literal(2);
    p.add_instruction(add{}, one, two);

The `add_literal <migraphx::program::add_literal>` function will add an instruction to the program to store a literal number. The `instruction_ref <migraphx::instruction_ref>` is a reference to the instruction in the program, which can be used to compose the output of the instruction with another instruction.

After creating the literals, we then create the instruction to add the numbers together. This is done by using the `add{} <migraphx::add>` operation class along with the `instruction_ref <migraphx::instruction_ref>` for the input arguments of the instruction.

Finally, we can run this `program <migraphx::program>` by compiling it for the cpu and then running it with `eval <migraphx::program::eval>`::

    p.compile(cpu::target{});
    argument result = p.eval({});

The easiest way to see the result is to print it::

    std::cout << result;

Which will print ``3``.

We can also compile the program for the gpu as well.

Adding parameters
-----------------

Of course, this program will always produce the same value which is quite uninteresting. Instead, we want to pass an input to a program and compute a value based on the input. This can be done with a parameter. For example, we can modify the program to take an input ``x``::

    program p;
    instruction_ref x = p.add_parameter("x", {shape::int64_type});
    instruction_ref two = p.add_literal(2);
    p.add_instruction(add{}, x, two);
    p.compile(cpu::target{});

This adds a parameter of type ``int64``, and compiles it for the ``cpu``. To run the program, we need to pass the parameter to it when we call `eval <migraphx::program::eval>`::

    argument result = p.eval({
        {"x", literal{1}.get_argument()}
    });
    std::cout << result;

This will print ``3``.

A parameter is given as an `argument <migraphx::argument>`. In this case, the simplest way of creating an `argument <migraphx::argument>` is from a `literal <migraphx::literal>`.

Tensor data
-----------

In this example we are just creating numbers, but the `shape <migraphx::shape>` class can describe multi-dimensional tensors. For example, we can build a simple network with convolution and relu::

    program p;
    instruction_ref input = p.add_parameter("x", shape{shape::float_type, {1, 3, 32, 32}});
    instruction_ref weights = p.add_parameter("w", shape{shape::float_type, {1, 3, 5, 5}});
    instruction_ref conv = p.add_instruction(convolution{}, input, weights);
    p.add_instruction(activation{"relu"}, conv);

Here we create two parameters for both the ``input`` and ``weights``. In the previous examples, we just created simple literals, however, most programs will take data from already allocated buffers(usually on the GPU). In this case, we can create `argument <migraphx::argument>` objects directly from the pointers to the buffers::

    // Compile the program
    p.compile(gpu::target{});
    // Allocated buffers by the user
    float* input = ...;
    float* weights = ...;
    // Create the arguments
    argument input_arg{shape{shape::float_type, {1, 3, 32, 32}}, input};
    argument weights_arg{shape{shape::float_type, {1, 3, 32, 32}}, weights};
    p.eval({{"x", input_arg}, {"w", weights_arg}})

An `argument <migraphx::argument>` can handle memory buffers from either the GPU or the CPU, but when running the `program <migraphx::program>`, buffers should be allocated for the corresponding target. That is, when compiling for the CPU, the buffers should be allocated on the CPU, and when compiling for the GPU the buffers should be allocated on the GPU.

Importing from onnx
-------------------

A `program <migraphx::program>` can be built directly from an onnx file, which makes it easier to use neural networks directly from other frameworks. In this case, there is an ``parse_onnx`` function::

    program p = parse_onnx("model.onnx");
    p.compile(gpu::target{});

