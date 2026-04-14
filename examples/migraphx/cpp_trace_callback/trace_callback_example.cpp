/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <migraphx/migraphx.hpp>

static void print_values(const migraphx::argument& arg, std::size_t max_elems = 8)
{
    auto shape = arg.get_shape();
    if(shape.type() == migraphx_shape_tuple_type)
    {
        std::cout << "(tuple)";
        return;
    }
    auto lens      = shape.lengths();
    std::size_t sz = 1;
    for(auto l : lens)
        sz *= l;
    auto n = std::min<std::size_t>(sz, max_elems);
    std::vector<float> data(n);
    std::memcpy(data.data(), arg.data(), n * sizeof(float));
    std::cout << "[";
    for(std::size_t i = 0; i < n; i++)
    {
        if(i > 0)
            std::cout << ", ";
        std::cout << data[i];
    }
    if(sz > max_elems)
        std::cout << ", ...";
    std::cout << "]";
}

static void print_cb(size_t idx, const char* name, const migraphx::argument& output)
{
    std::cout << "  @" << idx << " " << name << "\n    -> ";
    print_values(output);
    std::cout << "\n\n";
}

int main()
{
    //   branch_a = relu(x + y)
    //   branch_b = sigmoid(x * y)
    //   out      = concat(a, b)
    migraphx::program p;
    auto mm = p.get_main_module();
    migraphx::shape s(migraphx_shape_float_type, {2, 3});

    auto x = mm.add_parameter("x", s);
    auto y = mm.add_parameter("y", s);

    auto sum      = mm.add_instruction(migraphx::operation("add"), migraphx::instructions(x, y));
    auto branch_a = mm.add_instruction(migraphx::operation("relu"), migraphx::instructions(sum));

    auto prod = mm.add_instruction(migraphx::operation("mul"), migraphx::instructions(x, y));
    auto branch_b =
        mm.add_instruction(migraphx::operation("sigmoid"), migraphx::instructions(prod));

    mm.add_instruction(migraphx::operation("concat", "{axis: 1}"),
                       migraphx::instructions(branch_a, branch_b));

    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30, 40, 50, 60};
    migraphx::program_parameters params;
    params.add("x", migraphx::argument(s, x_data.data()));
    params.add("y", migraphx::argument(s, y_data.data()));

    // 1. Inspect every operator
    std::cout << "=== All operators ===\n";
    p.run_trace(params, [](size_t idx, const char* name, migraphx::argument output) {
        print_cb(idx, name, output);
    });

    // 2. Filter by name
    std::cout << "=== concat_kernel (by name) ===\n";
    p.run_trace(params, [](size_t idx, const char* name, migraphx::argument output) {
        if(std::string(name).find("concat_kernel") != std::string::npos)
            print_cb(idx, name, output);
    });

    // 3. Filter by instruction index
    std::cout << "=== concat_kernel (by index) ===\n";
    p.run_trace(params, [](size_t idx, const char* name, migraphx::argument output) {
        if(idx == 13)
            print_cb(idx, name, output);
    });
}
