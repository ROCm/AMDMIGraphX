/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_reduce{}, migraphx::dead_code_elimination{}});
}

template <class F>
migraphx::instruction_ref add_reduce(migraphx::program& p,
                                     const std::string& name,
                                     std::vector<migraphx::instruction_ref> inputs,
                                     const std::vector<int64_t>& axes,
                                     F f)
{
    auto* rm = p.create_module(name);
    auto* mm = p.get_main_module();
    rm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return rm->add_parameter(
            "x" + std::to_string(params.size()),
            migraphx::shape{input->get_shape().type(), input->get_shape().lens()});
    });
    auto r = f(rm, params, axes);
    rm->add_return({r});
    return mm->add_instruction(migraphx::make_op("fused_reduce", {{"axes", axes}}), inputs, {rm});
}

inline auto single_reduce(const std::string& name)
{
    return [=](auto* rm, const auto& inputs, const auto& axes) {
        return rm->add_instruction(migraphx::make_op(name, {{"axes", axes}}), inputs);
    };
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        mm->add_return({rsum1, rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {y}, {1}, single_reduce("reduce_sum"));
        mm->add_return({rsum1, rsum2});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
