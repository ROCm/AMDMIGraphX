/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

struct pre_gemm_softmax_gemm : migraphx::gpu::gemm_softmax_gemm
{
    std::string name() const { return "gpu::pre_gemm_softmax_gemm"; }
};

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::prefuse_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(find_gemm_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto z     = mm->add_parameter("z", s1);
        auto scale = mm->add_literal(2.0f);

        auto dot1     = mm->add_instruction(migraphx::make_op("dot"), x, y);
        auto scale_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), scale);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), dot1, scale_mb);
        auto sm   = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), mul);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), sm, z);
        mm->add_return({dot2});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto z   = mm->add_parameter("z", s1);

        auto attn =
            mm->add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 2}, x, y, z);

        mm->add_return({attn});
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2;
    if(migraphx::gpu::mlir_attention_enabled())
    {
        p2 = create_fused_program();
    }
    else
    {
        p2 = p1;
    }

    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(find_gemm_softmax_gemm_multi_scale)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto z   = mm->add_parameter("z", s1);
        auto scale =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}, 10));

        auto dot1     = mm->add_instruction(migraphx::make_op("dot"), x, y);
        auto scale_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), scale);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), dot1, scale_mb);
        auto sm   = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), mul);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), sm, z);
        mm->add_return({dot2});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto z   = mm->add_parameter("z", s1);

        auto attn =
            mm->add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 1}, x, y, z);

        mm->add_return({attn});
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2;
    if(migraphx::gpu::mlir_attention_enabled())
    {
        p2 = create_fused_program();
    }
    else
    {
        p2 = p1;
    }

    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(find_gemm_softmax_gemm_no_scale)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto z   = mm->add_parameter("z", s1);

        auto dot1     = mm->add_instruction(migraphx::make_op("dot"), x, y);
        auto sm   = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), dot1);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), sm, z);
        mm->add_return({dot2});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto z   = mm->add_parameter("z", s1);

        auto attn =
            mm->add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 1}, x, y, z);

        mm->add_return({attn});
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2;
    if(migraphx::gpu::mlir_attention_enabled())
    {
        p2 = create_fused_program();
    }
    else
    {
        p2 = p1;
    }

    run_pass(p1);

    EXPECT(p1 == p2);
}


int main(int argc, const char* argv[]) { test::run(argc, argv); }
