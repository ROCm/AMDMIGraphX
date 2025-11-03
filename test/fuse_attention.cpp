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
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/fuse_attention.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <basic_ops.hpp>
#include <group.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <utility>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_FLASH_DECODING);

static void run_pass(migraphx::program& p, migraphx::fuse_attention fa = {})
{
    migraphx::run_passes(p, {fa, migraphx::dead_code_elimination{}});
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
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());
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
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), where, rmax);
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
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p2,
            "attn0",
            "attention",
            {a, b, eight, select, ten, b1},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    gm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), where, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[5]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(gemm_multi_use_pw_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 16, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 4, 8, 16}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 4, 16, 16}};
    migraphx::shape s_mask{migraphx::shape::int64_type, {2, 16}};
    auto s1_elements = s1.elements();

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto mask = mm->add_parameter("mask", s_mask);
        auto x    = mm->add_parameter("x", s1);

        std::vector<float> c1_vec(s1_elements, 0.125);
        std::vector<float> c2_vec(s1_elements, 10);
        auto c1       = mm->add_literal(migraphx::literal(s2, c1_vec));
        auto c2       = mm->add_literal(migraphx::literal(s1, c2_vec));
        auto ten      = mm->add_literal(migraphx::literal(10.0f));
        auto zero     = mm->add_literal(migraphx::literal(0.0f));
        auto zero_int = mm->add_literal(migraphx::literal(0));
        auto scale    = mm->add_literal(migraphx::literal(0.25f));

        mask = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), mask);
        mask     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mask);
        zero_int = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", mask->get_shape().lens()}}),
            zero_int);
        auto eq = mm->add_instruction(migraphx::make_op("equal"), mask, zero_int);
        eq      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), eq);

        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), x, c1);

        eq =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), eq);
        ten  = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                  ten);
        zero = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                   zero);
        auto where = mm->add_instruction(migraphx::make_op("where"), eq, ten, zero);
        scale = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                    scale);
        auto add = mm->add_instruction(migraphx::make_op("add"), gemm1, where);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), add, scale);

        auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mul);
        rmax      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", mul->get_shape().lens()}}), rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), mul, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        rsum      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", mul->get_shape().lens()}}), rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, c2);
        mm->add_return({gemm2, zero, eq, scale});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto mask = mm->add_parameter("mask", s_mask);
        auto x    = mm->add_parameter("x", s1);

        std::vector<float> c1_vec(s1_elements, 0.125);
        std::vector<float> c2_vec(s1_elements, 10);
        auto c1       = mm->add_literal(migraphx::literal(s2, c1_vec));
        auto c2       = mm->add_literal(migraphx::literal(s1, c2_vec));
        auto ten      = mm->add_literal(migraphx::literal(10.0f));
        auto zero     = mm->add_literal(migraphx::literal(0.0f));
        auto zero_int = mm->add_literal(migraphx::literal(0));
        auto scale    = mm->add_literal(migraphx::literal(0.25f));

        mask = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), mask);
        mask     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mask);
        zero_int = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", mask->get_shape().lens()}}),
            zero_int);
        auto eq = mm->add_instruction(migraphx::make_op("equal"), mask, zero_int);
        eq      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), eq);

        eq =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), eq);
        ten  = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                  ten);
        zero = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                   zero);
        auto where = mm->add_instruction(migraphx::make_op("where"), eq, ten, zero);
        scale = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}),
                                    scale);

        auto group = add_group(
            p2, "attn0", "attention", {x, c1, where, scale, c2}, [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add   = gm->add_instruction(migraphx::make_op("add"), gemm1, inputs[2]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), add, inputs[3]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mul);
                rmax = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), mul, rmax);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                rsum = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[4]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group, zero, eq, scale});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(gemm_pw_softmax_lse_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    migraphx::shape s3{migraphx::shape::half_type, {1, 12, 256, 1}};
    auto s1_elements = s1.elements();
    auto s3_elements = s3.elements();

    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        std::vector<float> log2s(s3_elements, 1.44238);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        auto log2  = mm->add_literal(migraphx::literal{s3, log2s});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        auto gemm1   = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto mul     = mm->add_instruction(migraphx::make_op("mul"), gemm1, eight);
        auto where   = mm->add_instruction(migraphx::make_op("where"), select, mul, ten);
        auto rmax    = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
        auto rmax_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), where, rmax_mb);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto log  = mm->add_instruction(migraphx::make_op("log"), rsum);
        rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rsum);
        auto div        = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto adjust_lse = mm->add_instruction(migraphx::make_op("add"), log, rmax);
        auto log2se     = mm->add_instruction(migraphx::make_op("mul"), adjust_lse, log2);

        auto convert = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), log2se);
        auto lse = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), convert);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2, lse});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        std::vector<float> log2s(s3_elements, 1.44238);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        auto log2  = mm->add_literal(migraphx::literal{s3, log2s});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);

        auto group = add_group(
            p2,
            "attn0",
            "attention",
            {a, b, eight, select, ten, b1},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto mul   = gm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    gm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
                auto rmax_mb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), where, rmax_mb);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto rsum_mb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum_mb);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[5]);
                auto log   = gm->add_instruction(migraphx::make_op("log"), rsum);
                auto add   = gm->add_instruction(migraphx::make_op("add"), log, rmax);
                return std::vector<migraphx::instruction_ref>{gemm2, add};
            });

        auto adjust_lse =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), group);
        auto log2se = mm->add_instruction(migraphx::make_op("mul"), adjust_lse, log2);

        auto convert = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), log2se);
        auto lse = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), convert);
        auto gemm2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), group);

        mm->add_return({gemm2, lse});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(gemm_softmax_gemm_flash_decoding)
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
    run_pass(p1, {.flash_decoding_num_splits = 2});
    migraphx::program p2;
    {
        auto* mm         = p2.get_main_module();
        auto a           = mm->add_parameter("1", s1);
        auto b           = mm->add_parameter("2", s1);
        auto b1          = mm->add_parameter("3", s1);
        auto a_unsqueeze = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), a);
        auto a_broadcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 256}}}), a_unsqueeze);
        auto b_transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        // K needs reshape + transpose: [1, 12, 256, 256] -> [1, 12, 256, 2, 128] -> [1, 12, 2, 256, 128]
        auto b_reshape_intermediate = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 12, 256, 2, 128}}}), b_transpose);
        auto b_reshape = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2, 4}}}), b_reshape_intermediate);
        auto b1_transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b1);
        auto b1_reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 12, 2, 128, 256}}}), b1_transpose);
        auto group = add_group(
            p2,
            "attn0_flash_decoding",
            "attention",
            {a_broadcast, b_reshape, b1_reshape},
            {"x0", "x1", "x2"},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {4}}}), gemm1);
                auto rmax_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 128}}}),
                    rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax_broad);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {4}}}), exp);
                auto rsum_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 128}}}),
                    rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum_broad);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                auto log   = gm->add_instruction(migraphx::make_op("log"), rsum);
                auto add   = gm->add_instruction(migraphx::make_op("add"), rmax, log);
                return std::vector<migraphx::instruction_ref>{gemm2, add};
            });
        auto o_p = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), group);
        auto lse = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), group);
        auto k2_rmax   = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), lse);
        auto k2_broad1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 1}}}), k2_rmax);
        auto k2_sub = mm->add_instruction(migraphx::make_op("sub"), lse, k2_broad1);
        auto k2_exp = mm->add_instruction(migraphx::make_op("exp"), k2_sub);
        auto k2_rsum1 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), k2_exp);
        auto k2_broad2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 1}}}), k2_rsum1);
        auto k2_div    = mm->add_instruction(migraphx::make_op("div"), k2_exp, k2_broad2);
        auto k2_broad3 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 256, 256}}}), k2_div);
        auto k2_convert = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), k2_broad3);
        auto k2_mul = mm->add_instruction(migraphx::make_op("mul"), o_p, k2_convert);
        auto k2_rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), k2_mul);
        auto k2_squeeze =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), k2_rsum2);
        mm->add_return({k2_squeeze});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(flash_decoding_3d)
{
    // 3D Shape: [batch, sequence_length, head_dim]
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 256}};
    const std::size_t num_splits = 2;

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("q", s_3d);
        auto b   = mm->add_parameter("k", s_3d);
        auto b1  = mm->add_parameter("v", s_3d);
        b  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gemm1);
        rmax       = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
        rsum      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1, {.flash_decoding_num_splits = 2});

    migraphx::program p2;
    {
        auto* mm      = p2.get_main_module();
        auto a        = mm->add_parameter("q", s_3d);
        auto b        = mm->add_parameter("k", s_3d);
        auto b1       = mm->add_parameter("v", s_3d);
        size_t g_axis = 1;

        // New shapes for flash decoding
        std::vector<size_t> q_prime_shape = {1, num_splits, 256, 256};
        std::vector<size_t> k_prime_shape = {1, num_splits, 256, 128};
        std::vector<size_t> v_prime_shape = {1, num_splits, 128, 256};

        auto a_unsqueeze =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {g_axis}}}), a);
        auto a_broadcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", q_prime_shape}}), a_unsqueeze);

        auto b_transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
        // K needs reshape + transpose: [1, 256, 256] -> [1, 256, 2, 128] -> [1, 2, 256, 128]
        auto b_reshape_intermediate = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 256, 2, 128}}}), b_transpose);
        auto b_reshape = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), b_reshape_intermediate);

        auto b1_transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b1);
        auto b1_reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", v_prime_shape}}), b1_transpose);

        auto group = add_group(
            p2,
            "attn0_flash_decoding",
            "attention",
            {a_broadcast, b_reshape, b1_reshape},
            {"x0", "x1", "x2"},
            [&](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
                auto rmax_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, num_splits, 256, 128}}}),
                    rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax_broad);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto rsum_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, num_splits, 256, 128}}}),
                    rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum_broad);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                auto log   = gm->add_instruction(migraphx::make_op("log"), rsum);
                auto add   = gm->add_instruction(migraphx::make_op("add"), rmax, log);
                return std::vector<migraphx::instruction_ref>{gemm2, add};
            });
        auto o_p = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), group);
        auto lse = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), group);

        // Kernel 2
        auto k2_rmax =
            mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {g_axis}}}), lse);
        auto k2_broad1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, num_splits, 256, 1}}}), k2_rmax);
        auto k2_sub = mm->add_instruction(migraphx::make_op("sub"), lse, k2_broad1);
        auto k2_exp = mm->add_instruction(migraphx::make_op("exp"), k2_sub);
        auto k2_rsum1 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {g_axis}}}), k2_exp);
        auto k2_broad2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, num_splits, 256, 1}}}), k2_rsum1);
        auto k2_div    = mm->add_instruction(migraphx::make_op("div"), k2_exp, k2_broad2);
        auto k2_broad3 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", q_prime_shape}}), k2_div);
        auto k2_convert = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), k2_broad3);
        auto k2_mul = mm->add_instruction(migraphx::make_op("mul"), o_p, k2_convert);
        auto k2_rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {g_axis}}}), k2_mul);
        auto k2_squeeze =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {g_axis}}}), k2_rsum2);
        mm->add_return({k2_squeeze});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(flash_decoding_disabled)
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
    run_pass(p1, {.flash_decoding_num_splits = 0});

    // Expected result: only attention fusion, no flash decoding
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
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[])
{
    test::run(argc, argv);
    return 0;
}
