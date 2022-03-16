MIGraphX Fundamentals
========

MIGraphX provides an optimized execution engine for deep learning neural networks.
In this introduction for developers of MIGraphX, we will cover some simple operations in the MIGraphX framework.


Location of the Examples
------------------

The `ref_dev_examples.cpp` can be found in the test directory (`/test`).
The executable file `test_ref_dev_examples` based on this file will be created in the `bin/` of the build directory after running `make -j$(nproc) check`.


Adding Two Literals
------------------

A program consists of a set of instructions to be executed when calling `eval <migraphx::program::eval>`.
Each instruction has an associated `operation <migraphx::operation>` which represents the computation to be performed by the instruction.

We start a snippet of the simple `add_two_literals()` function::

    // create the program and get a pointer to the main module
    migraphx::program p;
    auto* mm = p.get_main_module();

    // add two literals to the program
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);

    // make the add operation between the two literals and add it to the program
    mm->add_instruction(migraphx::make_op("add"), one, two);

    // compile the program on the reference device
    p.compile(migraphx::ref::target{});

    // evaulate the program and retreive the result
    auto result = p.eval({}).back();
    std::cout << "add_two_literals: 1 + 2 = " << result << "\n";

We start by creating a simple `migraphx::program` object and then getting a pointer to the main module of it.
The program is split into ``modules``, so instructions are added to the modules rather than directly onto the program object.
We then use the `add_literal <migraphx::program::add_literal>` function to add an instruction that stores the literal number ``1`` while returning an `instruction_ref <migraphx::instruction_ref>`.
The returned `instruction_ref <migraphx::instruction_ref>` can be used in another instruction as an input.
We use the same `add_literal <migraphx::program::add_literal>` function to add a ``2`` to the program.
After creating the literals, we then create the instruction to add the numbers together.
This is done by using the `add_instruction <migraphx::program::add_instruction>` function with the ``"add"`` `operation <migraphx::program::operation>` created by `make_op <migraphx::program::make_op>` along with the previous `add_literal` `instruction_ref <migraphx::instruction_ref>` for the input arguments of the instruction.
Finally, we can run this `program <migraphx::program>` by compiling it for the reference target (CPU) and then running it with `eval <migraphx::program::eval>`
The result is then retreived and printed to the console.

We can compile the program for the gpu as well, but the file will have to be moved to the `test/gpu/` directory and the correct target must be included::

    #include <migraphx/gpu/target.hpp


Using Parameters
-----------------

The previous program will always produce the same value of adding ``1`` and ``2``.
In the next program we want to pass an input to a program and compute a value based on the input.
We can modify the program to take an input parameter ``x``, as seen in the ``add_parameter()`` function::

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1}};

    // add a "x" parameter with the shape s
    auto x   = mm->add_parameter("x", s);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), x, two);
    p.compile(migraphx::ref::target{});

This adds a parameter of type ``int32``, and compiles it for the CPU.
To run the program, we need to pass the parameter as a `parameter_map` when we call `eval <migraphx::program::eval>`.
We create the `parameter_map` by setting the ``x`` key to an `argument <migraphx::argument>` object with an ``int`` data type::

    std::vector<int> data = {4};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::cout << "add_parameters: 4 + 2 = " << result << "\n";
    EXPECT(result.at<int>() == 6);


Handling Tensor Data
-----------

In the previous examples we have only been dealing with scalars, but the `shape <migraphx::shape>` class can describe multi-dimensional tensors.
For example, we can compute a simple convolution::

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), input, weights);

Here we create two parameters for both the ``input`` and ``weights``.
In the previous examples, we just created simple literals, however, most programs will take data from already allocated buffers (usually on the GPU).
In this case, we can create `argument <migraphx::argument>` objects directly from the pointers to the buffers::

    // Compile the program
    p.compile(migraphx::ref::target{});

    // Allocated buffers by the user
    std::vector<float> a = ...;
    std::vector<float> c = ...;

    // Solution vector
    std::vector<float> sol = ...;

    // Create the arguments in a parameter_map
    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(input_shape, a.data());
    pp["W"] = migraphx::argument(weights_shape, c.data());

    // Evaluate and confirm the result
    auto result = p.eval(pp).back();
    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(results_vector, sol));

An `argument <migraphx::argument>` can handle memory buffers from either the GPU or the CPU, but when running the `program <migraphx::program>`, buffers should be allocated for the corresponding target.
That is, when compiling for the CPU, the buffers should be allocated on the CPU, and when compiling for the GPU the buffers should be allocated on the GPU.


Importing From ONNX
-------------------

A `program <migraphx::program>` can be built directly from an onnx file using the MIGraphX ONNX parser.
This makes it easier to use neural networks directly from other frameworks.
In this case, there is an ``parse_onnx`` function::

    program p = migraphx::parse_onnx("model.onnx");
    p.compile(migraphx::gpu::target{});

