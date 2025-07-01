/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "add_group_op.hpp"
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_special_ops.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <utility>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_special_ops{.enable_attention = true}});
}

TEST_CASE(gemm_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
        rmax = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        auto group = add_group(
            p2,
            "attn0",
            "attention",
            {a, b, b1},
            {"x0", "x1", "x2"},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div = gm->add_instruction(migraphx::make_op("div"), exp, rsum);

                return gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
            });
        mm->add_return({group});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(gemm_pw_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();

    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto mul   = mm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
        auto where = mm->add_instruction(migraphx::make_op("where"), select, mul, ten);
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
        rmax = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1);

    // Same result as gemm_softmax_gemm, but here the fused_reduce is unrolled into
    // pointwise ops + softmax
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p2, "attn0", "attention", {a, b, select, b1}, [=](auto* gm, const auto& inputs) {
                auto ten   = gm->add_literal(migraphx::literal{s1, tens});
                auto eight = gm->add_literal(migraphx::literal{s1, eights});
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
                auto where = gm->add_instruction(migraphx::make_op("where"), inputs[2], mul, ten);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div = gm->add_instruction(migraphx::make_op("div"), exp, rsum);

                return gm->add_instruction(migraphx::make_op("dot"), div, inputs[3]);
            });
        mm->add_return({group});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[])
{
    test::run(argc, argv);
    return 0;
}
