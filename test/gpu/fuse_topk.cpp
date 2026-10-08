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
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <test.hpp>

// The kernels launched by the program, without the output copies
static std::vector<std::string> get_kernel_names(migraphx::program& p)
{
    std::vector<std::string> result;
    auto im = migraphx::iterator_for(*p.get_main_module());
    migraphx::transform_if(
        im.begin(),
        im.end(),
        std::back_inserter(result),
        [](auto ins) {
            if(ins->name() != "gpu::code_object")
                return false;
            const auto& op = migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator());
            return op.symbol_name != "hip_copy_kernel";
        },
        [](auto ins) {
            return migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator())
                .symbol_name;
        });
    return result;
}

// MoE router: softmax -> topk -> values / sum(values) compiles to a single kernel
TEST_CASE(softmax_topk_normalize_single_kernel)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 128}};
    auto x       = mm->add_parameter("x", s);
    auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    auto topk    = mm->add_instruction(migraphx::make_op("topk", {{"axis", 1}, {"k", 4}}), softmax);
    auto values  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    auto sum     = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), values);
    auto sumb    = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", values->get_shape().lens()}}), sum);
    auto norm = mm->add_instruction(migraphx::make_op("div"), values, sumb);
    mm->add_return({norm, indices});
    p.compile(migraphx::make_target("gpu"));

    auto kernels = get_kernel_names(p);
    EXPECT(kernels.size() == 1);
    EXPECT(migraphx::contains(kernels.front(), "topk"));
    EXPECT(migraphx::contains(kernels.front(), "reduce_sum"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
