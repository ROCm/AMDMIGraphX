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

// Demonstrates using eval_callback to inspect operator output buffers during
// GPU program evaluation.  Build with cmake (see CMakeLists.txt).

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <migraphx/program.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/eval_callback.hpp>
#include <migraphx/iterator_for.hpp>

using ins_name_map = std::unordered_map<migraphx::instruction_ref, std::string>;

static std::string extract_symbol_name(const migraphx::operation& op)
{
    std::stringstream ss;
    ss << op;
    auto full = ss.str();
    auto pos  = full.find("symbol_name=");
    if(pos == std::string::npos)
        return "";
    pos += 12;
    auto end = full.find_first_of(",]", pos);
    return full.substr(pos, end - pos);
}

static ins_name_map build_instruction_names(const migraphx::program& p)
{
    ins_name_map result;
    p.print([&](auto ins, const auto& names) {
        auto sym    = extract_symbol_name(ins->get_operator());
        result[ins] = names.at(ins) + " = " + (sym.empty() ? ins->name() : sym);
    });
    return result;
}

static void print_values(const migraphx::argument& output, std::size_t max_elems = 8)
{
    if(output.get_shape().type() == migraphx::shape::tuple_type)
    {
        std::cout << "(tuple with " << output.get_sub_objects().size() << " elements)";
        return;
    }
    output.visit([&](auto v) {
        auto n = std::min<std::size_t>(v.size(), max_elems);
        std::cout << "[";
        for(std::size_t i = 0; i < n; ++i)
        {
            if(i > 0)
                std::cout << ", ";
            std::cout << v[i];
        }
        if(v.size() > max_elems)
            std::cout << ", ...";
        std::cout << "]";
    });
}

int main()
{
    // ---------------------------------------------------------------
    // Two independent branches joined by concat -- the compiler
    // cannot fuse across a concat boundary, producing separate
    // gpu::code_object kernels:
    //
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

    // Build a map of instruction_ref -> readable label using the same
    // mechanism as program::print() / MIGRAPHX_TRACE_EVAL.  Each entry
    // looks like: "@9 =
    // gpu::code_object[...,symbol_name=add_relu_mul_sigmoid_kernel,...](@7,@4,@8)"
    auto names = build_instruction_names(p);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30, 40, 50, 60};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, x_data.data());
    params["y"] = migraphx::argument(s, y_data.data());

    // ---------------------------------------------------------------
    // 1.  Inspect every operator with readable labels
    // ---------------------------------------------------------------
    std::cout << "=== All operators ===\n";
    migraphx::eval_callback cb_all(
        [&](migraphx::instruction_ref ins, const migraphx::argument& output) {
            std::cout << "  " << names.at(ins) << "\n";
            std::cout << "    -> ";
            print_values(output);
            std::cout << "\n\n";
        });
    p.eval(params, cb_all);

    // ---------------------------------------------------------------
    // 2.  Filter by operator name -- only gpu::code_object kernels
    // ---------------------------------------------------------------
    std::cout << "=== Only gpu::code_object (by name) ===\n";
    migraphx::eval_callback cb_name(
        [&](migraphx::instruction_ref ins, const migraphx::argument& output) {
            std::cout << "  " << names.at(ins) << "\n";
            std::cout << "    -> ";
            print_values(output);
            std::cout << "\n\n";
        },
        {"gpu::code_object"});
    p.eval(params, cb_name);

    // ---------------------------------------------------------------
    // 3.  Filter by instruction_ref -- find the concat kernel in the
    //     compiled graph and target it specifically.
    // ---------------------------------------------------------------
    std::cout << "=== Only the concat kernel (by ref, post-compile) ===\n";
    const auto* compiled_mm              = p.get_main_module();
    migraphx::instruction_ref concat_ins = compiled_mm->end();
    for(auto ins : migraphx::iterator_for(*compiled_mm))
    {
        if(ins->name() == "gpu::code_object" and
           ins->get_shape().type() != migraphx::shape::tuple_type)
        {
            concat_ins = ins;
        }
    }

    if(concat_ins != compiled_mm->end())
    {
        migraphx::eval_callback cb_ref(
            [&](migraphx::instruction_ref ins, const migraphx::argument& output) {
                std::cout << "  " << names.at(ins) << "\n";
                std::cout << "    -> ";
                print_values(output);
                std::cout << "\n\n";
            },
            std::unordered_set<migraphx::instruction_ref>{concat_ins});
        p.eval(params, cb_ref);
    }
}
