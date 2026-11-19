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

#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::dead_code_elimination{}});
}

// The submodule parameter is standard and the input is a transposed view: lowering copies the
// input into the parameter layout, since the compiled submodule reads it in that layout
TEST_CASE(select_module_lowering_copies_input_to_parameter_layout)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p;
    auto* sub  = p.create_module("sub");
    auto sdata = sub->add_parameter("data", s);
    sub->add_return({sub->add_instruction(migraphx::make_op("neg"), sdata)});

    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    auto transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{s}})}}),
        {transposed},
        {sub});
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    run_lowering(*mm);

    auto selections = std::count_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    EXPECT(selections == 1);
    auto it = std::find_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    if(it == mm->end())
        return;
    auto input = it->inputs().front();
    EXPECT(input->name() == "gpu::contiguous");
    EXPECT(input->get_shape().standard());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
