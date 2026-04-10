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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/eval_callback.hpp>

static std::string get_label(migraphx::instruction_ref ins)
{
    auto sym = migraphx::eval_callback::get_symbol_name(ins->get_operator());
    return sym.empty() ? ins->name() : sym;
}

static void print_flat(const migraphx::argument& arg, std::size_t max_elems = 8)
{
    arg.visit([&](auto v) {
        auto n = std::min<std::size_t>(v.size(), max_elems);
        std::cout << "[";
        const char* sep = "";
        std::for_each(v.begin(), v.begin() + n, [&](auto x) {
            std::cout << sep << x;
            sep = ", ";
        });
        if(v.size() > max_elems)
            std::cout << ", ...";
        std::cout << "]";
    });
}

static void print_values(const migraphx::argument& output, std::size_t max_elems = 8)
{
    if(output.get_shape().type() != migraphx::shape::tuple_type)
        return print_flat(output, max_elems);

    auto subs = output.get_sub_objects();
    std::cout << "(";
    const char* sep = "";
    std::for_each(subs.begin(), subs.end(), [&](const auto& sub) {
        std::cout << sep;
        print_flat(sub, max_elems);
        sep = ", ";
    });
    std::cout << ")";
}

int main()
{
    // ---------------------------------------------------------------
    //   branch_a = relu(x + y)
    //   branch_b = sigmoid(x * y)
    //   out      = concat(a, b)
    // ---------------------------------------------------------------
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);

    auto sum      = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto branch_a = mm->add_instruction(migraphx::make_op("relu"), sum);

    auto prod     = mm->add_instruction(migraphx::make_op("mul"), x, y);
    auto branch_b = mm->add_instruction(migraphx::make_op("sigmoid"), prod);

    mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), branch_a, branch_b);

    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30, 40, 50, 60};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, x_data.data());
    params["y"] = migraphx::argument(s, y_data.data());

    auto print_cb = [](migraphx::instruction_ref ins, const migraphx::argument& output) {
        std::cout << "  " << get_label(ins) << "\n    -> ";
        print_values(output);
        std::cout << "\n\n";
    };

    // 1. Inspect every operator
    std::cout << "=== All operators ===\n";
    p.eval(params, migraphx::eval_callback(print_cb));

    // 2. Filter by symbol name
    std::cout << "=== concat_kernel (by name) ===\n";
    p.eval(params, migraphx::eval_callback(print_cb, {"concat_kernel"}));

    // 3. Filter by instruction index (@13 from the compiled graph)
    std::cout << "=== concat_kernel (by index) ===\n";
    auto ins_13 = std::next(p.get_main_module()->begin(), 13);
    p.eval(params, migraphx::eval_callback(print_cb, {}, {ins_13}));
}
