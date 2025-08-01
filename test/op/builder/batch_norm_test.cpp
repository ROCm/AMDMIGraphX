/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/common.hpp>
#include <test.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/builder/insert.hpp>

TEST_CASE(batch_norm_1d_op_builder_test)
{
    migraphx::program prog_op_builded = migraphx::program();
    migraphx::program prog_manually_built = migraphx::program();
    auto* mm_op = prog_op_builded.get_main_module();
    auto* mm    = prog_manually_built.get_main_module();

    auto x     = mm->add_parameter("x", {migraphx::shape::half_type, {2, 3, 4}});
    auto scale = mm->add_parameter("scale", {migraphx::shape::float_type, {3}});
    auto bias  = mm->add_parameter("bias", {migraphx::shape::float_type, {3}});
    auto mean  = mm->add_parameter("mean", {migraphx::shape::float_type, {3}});
    auto var   = mm->add_parameter("variance", {migraphx::shape::float_type, {3}});
  
    // copy input parameters to the module that'll be built by op-builder
    *mm_op = *mm;

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape::half_type, {1e-5f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bias);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    // obtain arguments
    const auto& from = mm_op->get_parameters();
    std::vector<migraphx::instruction_ref> args{from.rbegin(), from.rend()};
    
    // call the SUT
    migraphx::op::builder::add("batchnorm", *mm_op, args, {});

    EXPECT(prog_op_builded == prog_manually_built);
}

TEST_CASE(batch_norm_2d_op_builder_test)
{
    migraphx::program prog_op_builded = migraphx::program();
    migraphx::program prog_manually_built = migraphx::program();
    auto* mm_op = prog_op_builded.get_main_module();
    auto* mm    = prog_manually_built.get_main_module();

    auto x     = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3, 4, 4}});
    auto scale = mm->add_parameter("scale", {migraphx::shape::float_type, {3}});
    auto bias  = mm->add_parameter("bias", {migraphx::shape::float_type, {3}});
    auto mean  = mm->add_parameter("mean", {migraphx::shape::float_type, {3}});
    auto var   = mm->add_parameter("variance", {migraphx::shape::float_type, {3}});

    // copy input parameters to the module that'll be built by op-builder
    *mm_op = *mm;

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1e-5f}});

    auto usq_scale = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
    auto usq_bias  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
    auto usq_mean  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mean);
    auto usq_var   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), var);

    auto x_sub_mean = add_common_op(*mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(*mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(*mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(*mm, migraphx::make_op("add"), {r0, usq_bias});

    // obtain arguments
    const auto& from = mm_op->get_parameters();
    std::vector<migraphx::instruction_ref> args{from.rbegin(), from.rend()};
    
    // call the SUT
    migraphx::op::builder::add("batchnorm", *mm_op, args, {});

    EXPECT(prog_op_builded == prog_manually_built);
}
