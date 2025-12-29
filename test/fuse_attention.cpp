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
#include <migraphx/split_factor.hpp>
#include <migraphx/generic_float.hpp>
#include <migraphx/env.hpp>
#include <basic_ops.hpp>
#include <group.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <utility>
#include <cstdlib>

static void run_pass(migraphx::program& p, migraphx::fuse_attention fa = {})
{
    migraphx::run_passes(p, {fa, migraphx::dead_code_elimination{}});
}

// Test helper functions used in fuse_attention pass
TEST_CASE(get_num_splits_from_member)
{
    // Test that member variable takes precedence over environment variable
    migraphx::fuse_attention fa;
    fa.flash_decoding_num_splits = 8;

    // Test that struct members are set correctly
    EXPECT(fa.flash_decoding_num_splits == 8);
    EXPECT(fa.flash_decoding_threshold == 32);      // default value
    EXPECT(fa.flash_decoding_max_splits == 16);     // default value
    EXPECT(fa.flash_decoding_min_chunk_size == 32); // default value
}

TEST_CASE(calculate_flash_decoding_splits_basic)
{
    // sequence_length that can be split evenly
    // 256 with min_chunk=32 should split to 8 (256/8 = 32)
    std::size_t seq_len1 = 256;
    std::size_t result1  = migraphx::split_dim(seq_len1, 32, 16);
    EXPECT(result1 == 8);

    // sequence_length with max_splits constraint
    // 1024 with min_chunk=64 and max_splits=8 should be limited to 8
    std::size_t seq_len2 = 1024;
    std::size_t result2  = migraphx::split_dim(seq_len2, 64, 8);
    EXPECT(result2 == 8);

    // small sequence that shouldn't be split
    // 32 with min_chunk=32 should return 1 (no split)
    std::size_t seq_len3 = 32;
    std::size_t result3  = migraphx::split_dim(seq_len3, 32, 16);
    EXPECT(result3 == 1);

    // prime number sequence length
    // 97 with min_chunk=10 should return 1 (can't split prime)
    std::size_t seq_len4 = 97;
    std::size_t result4  = migraphx::split_dim(seq_len4, 10, 16);
    EXPECT(result4 == 1);

    // typical attention sequence lengths
    // 2048 with min_chunk=128 and max_splits=16
    std::size_t seq_len5 = 2048;
    std::size_t result5  = migraphx::split_dim(seq_len5, 128, 16);
    EXPECT(result5 == 16);
}

TEST_CASE(padding_calculation)
{
    // not evenly divisible - padding needed
    std::size_t seq_len2 = 100;
    std::size_t groups2  = 8;
    // 100 % 8 != 0, so padding is needed
    std::size_t padding2 = migraphx::ceil_mul_of(seq_len2, groups2) - seq_len2;
    EXPECT(padding2 == 4); // 104 - 100 = 4

    // sequence length = 127, groups = 16
    std::size_t seq_len3 = 127;
    std::size_t groups3  = 16;
    // 127 % 16 != 0, so padding is needed
    std::size_t padding3 = migraphx::ceil_mul_of(seq_len3, groups3) - seq_len3;
    EXPECT(padding3 == 1); // 128 - 127 = 1

    // large sequence with padding
    std::size_t seq_len4 = 2049;
    std::size_t groups4  = 32;
    // 2049 % 32 != 0, so padding is needed
    std::size_t padding4 = migraphx::ceil_mul_of(seq_len4, groups4) - seq_len4;
    EXPECT(padding4 == 31); // 2080 - 2049 = 31
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
    run_pass(p1, {.attn_enabled = true});

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
    run_pass(p1, {.attn_enabled = true});

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
    run_pass(p1, {.attn_enabled = true});

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
    run_pass(p1, {.attn_enabled = true});

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
    run_pass(
        p1, {.attn_enabled = true, .flash_decoding_enabled = true, .flash_decoding_num_splits = 2});
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
        // K: [1, 12, 256, 256] -> [1, 12, 256, 2, 128] -> [1, 12, 2, 256, 128]
        auto b_reshape_intermediate = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 12, 256, 2, 128}}}), b_transpose);
        auto b_reshape =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2, 4}}}),
                                b_reshape_intermediate);
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
    run_pass(
        p1, {.attn_enabled = true, .flash_decoding_enabled = true, .flash_decoding_num_splits = 2});

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
        // K: [1, 256, 256] -> [1, 256, 2, 128] -> [1, 2, 256, 128]
        auto b_reshape_intermediate = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 256, 2, 128}}}), b_transpose);
        auto b_reshape =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                b_reshape_intermediate);

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

TEST_CASE(kv_cache_attention)
{
    migraphx::shape s1{migraphx::shape::half_type, {1}};
    migraphx::shape s2{migraphx::shape::int32_type, {4}};
    migraphx::shape s3{migraphx::shape::half_type, {4, 1}};
    migraphx::shape s4{migraphx::shape::int32_type, {2, 1}};
    migraphx::shape s5{migraphx::shape::half_type, {2, 2, 4, 2}};
    migraphx::shape s6{migraphx::shape::half_type, {2, 1, 12}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto half = mm->add_literal(migraphx::literal{s1, {0.5}});
        auto ninf =
            mm->add_literal(migraphx::literal{s1, {-std::numeric_limits<float>::infinity()}});
        auto range     = mm->add_literal(migraphx::literal{s2, {1, 2, 3, 4}});
        auto sin_cache = mm->add_parameter("sin_cache", s3);
        auto cos_cache = mm->add_parameter("cos_cache", s3);
        auto slk       = mm->add_parameter("slk", s4);
        auto v         = mm->add_parameter("v", s5);
        auto k         = mm->add_parameter("k", s5);
        auto query     = mm->add_parameter("query", s6);
        auto rsp_q =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 6, 2}}}), query);
        auto tsp_q = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsp_q);
        auto rope = mm->add_instruction(
            migraphx::make_op("gqa_rotary_embedding",
                              {{"num_heads", 2}, {"kv_num_heads", 2}, {"interleaved", 0}}),
            tsp_q,
            slk,
            cos_cache,
            sin_cache);
        auto slc_k = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), rope);
        auto slc_v = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {6}}}), rope);
        auto cpp_k = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), slc_k, slk, k);
        auto cpp_v = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), slc_v, slk, v);
        auto slc_q = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), rope);
        auto tsp_k = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), cpp_k);
        auto gemm1    = mm->add_instruction(migraphx::make_op("dot"), slc_q, tsp_k);
        auto bc_range = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4}}}), range);
        auto bc_ninf = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), ninf);
        auto bc_half = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), half);
        auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, bc_half);
        auto bc_slk =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), slk);
        auto grtr      = mm->add_instruction(migraphx::make_op("greater"), bc_range, bc_slk);
        auto conv_grtr = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), grtr);
        auto unsq_grtr = mm->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {1, 2}}, {"steps", {}}}), conv_grtr);
        auto bc_grtr = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), unsq_grtr);
        auto mask      = mm->add_instruction(migraphx::make_op("where"), bc_grtr, bc_ninf, scaled);
        auto conv_mask = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mask);
        auto rdc_max =
            mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), conv_mask);
        auto bc_rm = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), rdc_max);
        auto sub     = mm->add_instruction(migraphx::make_op("sub"), conv_mask, bc_rm);
        auto exp     = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rdc_sum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto bc_rs   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), rdc_sum);
        auto div     = mm->add_instruction(migraphx::make_op("div"), exp, bc_rs);
        auto conv_sm = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto gemm2   = mm->add_instruction(migraphx::make_op("dot"), conv_sm, cpp_v);
        auto tsp_out = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm2);
        auto rsp_out =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 4}}}), tsp_out);
        mm->add_return({rsp_out, cpp_k, cpp_v});
    }
    run_pass(p1, {.attn_enabled = true});

    migraphx::program p2;
    {
        auto* mm       = p2.get_main_module();
        auto sin_cache = mm->add_parameter("sin_cache", s3);
        auto cos_cache = mm->add_parameter("cos_cache", s3);
        auto slk       = mm->add_parameter("slk", s4);
        auto v         = mm->add_parameter("v", s5);
        auto k         = mm->add_parameter("k", s5);
        auto query     = mm->add_parameter("query", s6);
        auto rsp_q =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 6, 2}}}), query);
        auto tsp_q = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsp_q);
        auto rope = mm->add_instruction(
            migraphx::make_op("gqa_rotary_embedding",
                              {{"num_heads", 2}, {"kv_num_heads", 2}, {"interleaved", 0}}),
            tsp_q,
            slk,
            cos_cache,
            sin_cache);
        auto slc_k = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), rope);
        auto slc_v = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {6}}}), rope);
        auto cpp_k = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), slc_k, slk, k);
        auto cpp_v = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), slc_v, slk, v);
        auto group = add_group(
            p2,
            "attn0",
            "kv_cache_attention",
            {rope, cpp_k, slk, cpp_v},
            [=](auto* gm, const auto& inputs) {
                auto half = gm->add_literal(migraphx::literal{s1, {0.5}});
                auto ninf = gm->add_literal(
                    migraphx::literal{s1, {-std::numeric_limits<float>::infinity()}});
                auto range = gm->add_literal(migraphx::literal{s2, {1, 2, 3, 4}});
                auto slc_q = gm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}),
                    inputs.at(0));
                auto tsp_k = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs.at(1));
                auto gemm1    = gm->add_instruction(migraphx::make_op("dot"), slc_q, tsp_k);
                auto bc_range = gm->add_instruction(
                    migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4}}}), range);
                auto bc_ninf = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), ninf);
                auto bc_half = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), half);
                auto scaled = gm->add_instruction(migraphx::make_op("mul"), gemm1, bc_half);
                auto bc_slk = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), inputs.at(2));
                auto grtr = gm->add_instruction(migraphx::make_op("greater"), bc_range, bc_slk);
                auto conv_grtr = gm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
                    grtr);
                auto unsq_grtr = gm->add_instruction(
                    migraphx::make_op("unsqueeze", {{"axes", {1, 2}}, {"steps", {}}}), conv_grtr);
                auto bc_grtr = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), unsq_grtr);
                auto mask =
                    gm->add_instruction(migraphx::make_op("where"), bc_grtr, bc_ninf, scaled);
                auto conv_mask = gm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
                    mask);
                auto rdc_max = gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}),
                                                   conv_mask);
                auto bc_rm   = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), rdc_max);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), conv_mask, bc_rm);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rdc_sum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto bc_rs = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 4}}}), rdc_sum);
                auto div     = gm->add_instruction(migraphx::make_op("div"), exp, bc_rs);
                auto conv_sm = gm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                    div);
                auto gemm2   = gm->add_instruction(migraphx::make_op("dot"), conv_sm, inputs.at(3));
                auto tsp_out = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm2);
                auto rsp_out = gm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 1, 4}}}), tsp_out);
                return std::vector<migraphx::instruction_ref>{rsp_out};
            });
        mm->add_return({group, cpp_k, cpp_v});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Test automatic splitting with num_splits = 0 (auto-calculate)
TEST_CASE(flash_decoding_3d_auto_split_large_sequence)
{
    // 3D Shape: [batch, sequence_length, head_dim] - Use larger sequence to trigger auto-splitting
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 512, 512}};

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
    // Use auto-splitting: num_splits = 0, with sequence length 512 > threshold 32
    run_pass(
        p1, {.attn_enabled = true, .flash_decoding_enabled = true, .flash_decoding_num_splits = 0});

    // Expected program with automatic splitting (should calculate 16 splits for 512 sequence)
    const std::size_t expected_splits = 16; // 512 = 2^9, split until chunk = 32, so 512/16 = 32
    migraphx::program p2;
    {
        auto* mm      = p2.get_main_module();
        auto a        = mm->add_parameter("q", s_3d);
        auto b        = mm->add_parameter("k", s_3d);
        auto b1       = mm->add_parameter("v", s_3d);
        size_t g_axis = 1;

        // New shapes for flash decoding with calculated splits
        std::vector<size_t> q_prime_shape = {1, expected_splits, 512, 512};
        std::vector<size_t> k_prime_shape = {1, expected_splits, 512, 32}; // 512/16 = 32
        std::vector<size_t> v_prime_shape = {1, expected_splits, 32, 512};

        auto a_unsqueeze =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {g_axis}}}), a);
        auto a_broadcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", q_prime_shape}}), a_unsqueeze);

        auto b_transpose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
        // K: [1, 512, 512] -> [1, 512, 16, 32] -> [1, 16, 512, 32]
        auto b_reshape_intermediate = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 512, expected_splits, 32}}}), b_transpose);
        auto b_reshape =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                b_reshape_intermediate);

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
                    migraphx::make_op("multibroadcast",
                                      {{"out_lens", {1, expected_splits, 512, 32}}}),
                    rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax_broad);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                auto rsum_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast",
                                      {{"out_lens", {1, expected_splits, 512, 32}}}),
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
            migraphx::make_op("multibroadcast", {{"out_lens", {1, expected_splits, 512, 1}}}),
            k2_rmax);
        auto k2_sub = mm->add_instruction(migraphx::make_op("sub"), lse, k2_broad1);
        auto k2_exp = mm->add_instruction(migraphx::make_op("exp"), k2_sub);
        auto k2_rsum1 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {g_axis}}}), k2_exp);
        auto k2_broad2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, expected_splits, 512, 1}}}),
            k2_rsum1);
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

TEST_CASE(flash_decoding_3d_auto_split_small_sequence)
{
    // 3D Shape: [batch, sequence_length, head_dim] - Small sequence that should NOT trigger
    // splitting
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 16, 16}};

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
    // Use auto-splitting: num_splits = 0, with small sequence length 16 < threshold 32
    run_pass(
        p1, {.attn_enabled = true, .flash_decoding_enabled = true, .flash_decoding_num_splits = 0});

    // Expected program with regular attention (no flash decoding for small sequence)
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("q", s_3d);
        auto b   = mm->add_parameter("k", s_3d);
        auto b1  = mm->add_parameter("v", s_3d);
        b  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b1);
        auto group = add_group(
            p2,
            "attn0",
            "attention",
            {a, b, b1},
            {"x0", "x1", "x2"},
            [=](auto* gm, const auto& inputs) {
                auto gemm1 = gm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto rmax =
                    gm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gemm1);
                auto rmax_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
                    rmax);
                auto sub = gm->add_instruction(migraphx::make_op("sub"), gemm1, rmax_broad);
                auto exp = gm->add_instruction(migraphx::make_op("exp"), sub);
                auto rsum =
                    gm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
                auto rsum_broad = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
                    rsum);
                auto div   = gm->add_instruction(migraphx::make_op("div"), exp, rsum_broad);
                auto gemm2 = gm->add_instruction(migraphx::make_op("dot"), div, inputs[2]);
                return std::vector<migraphx::instruction_ref>{gemm2};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(flash_decoding_4d_auto_split_custom_params)
{
    // 4D Shape: [batch, heads, sequence_length, head_dim] - Test with custom parameters
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
    // Test with custom min_chunk_size and max_splits
    run_pass(p1,
             {.attn_enabled                  = true,
              .flash_decoding_enabled        = true,
              .flash_decoding_num_splits     = 0, // Auto-calculate
              .flash_decoding_threshold      = 32,
              .flash_decoding_max_splits     = 4,    // Smaller max splits
              .flash_decoding_min_chunk_size = 64}); // Larger chunk size

    // Check for flash decoding
    bool found_flash_decoding = false;
    for(auto ins : *p1.get_main_module())
    {
        if(ins.name().find("group") != std::string::npos)
        {
            found_flash_decoding = true;
            break;
        }
    }

    EXPECT(found_flash_decoding);
}

TEST_CASE(flash_decoding_auto_split_threshold_behavior)
{
    // Test threshold behavior - sequence right at the threshold boundary
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 127, 127}};

    migraphx::program p1, p2;

    // Test 1: sequence length below threshold - should NOT split
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
    run_pass(p1,
             {.attn_enabled                  = true,
              .flash_decoding_enabled        = true,
              .flash_decoding_num_splits     = 0,
              .flash_decoding_threshold      = 128, // Greater than sequence length (127)
              .flash_decoding_max_splits     = 8,
              .flash_decoding_min_chunk_size = 32});

    // Test 2: sequence length at threshold - should split
    migraphx::shape s_3d_larger{migraphx::shape::half_type, {1, 128, 128}};
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("q", s_3d_larger);
        auto b   = mm->add_parameter("k", s_3d_larger);
        auto b1  = mm->add_parameter("v", s_3d_larger);
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
    run_pass(p2,
             {.attn_enabled                  = true,
              .flash_decoding_enabled        = true,
              .flash_decoding_num_splits     = 0,
              .flash_decoding_threshold      = 128, // Equal to sequence length (128)
              .flash_decoding_max_splits     = 8,
              .flash_decoding_min_chunk_size = 32});

    // Check results - look for flash decoding by checking module names
    bool found_flash_decoding_p1 = false, found_flash_decoding_p2 = false;
    bool found_regular_attention_p1 = false;

    for(auto ins : *p1.get_main_module())
    {
        if(ins.name().find("group") != std::string::npos)
        {
            // Check the module name to distinguish flash decoding from regular attention
            auto module_inputs = ins.module_inputs();
            if(!module_inputs.empty())
            {
                auto mod_name = module_inputs[0]->name();
                if(mod_name.find("flash_decoding") != std::string::npos)
                {
                    found_flash_decoding_p1 = true;
                }
                else
                {
                    found_regular_attention_p1 = true;
                }
            }
        }
    }

    for(auto ins : *p2.get_main_module())
    {
        if(ins.name().find("group") != std::string::npos)
        {
            auto module_inputs = ins.module_inputs();
            if(!module_inputs.empty())
            {
                auto mod_name = module_inputs[0]->name();
                if(mod_name.find("flash_decoding") != std::string::npos)
                {
                    found_flash_decoding_p2 = true;
                }
            }
        }
    }

    // Below threshold: should have regular attention, not flash decoding
    EXPECT(not found_flash_decoding_p1);
    EXPECT(found_regular_attention_p1); // Should have regular attention instead
    // At threshold: should have flash decoding
    EXPECT(found_flash_decoding_p2);
}

TEST_CASE(flash_decoding_auto_split_max_splits_constraint)
{
    // Test that max_splits constraint is respected
    migraphx::shape s_3d{migraphx::shape::half_type, {1, 2048, 2048}};

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
    // Use small max_splits to test constraint
    run_pass(p1,
             {.attn_enabled                  = true,
              .flash_decoding_enabled        = true,
              .flash_decoding_num_splits     = 0, // Auto-calculate
              .flash_decoding_threshold      = 32,
              .flash_decoding_max_splits     = 4, // Small max_splits
              .flash_decoding_min_chunk_size = 64});

    // Check that flash decoding was applied
    bool found_flash_decoding = false;
    for(auto ins : *p1.get_main_module())
    {
        if(ins.name().find("group") != std::string::npos)
        {
            found_flash_decoding = true;
            break;
        }
    }

    EXPECT(found_flash_decoding);
}

int main(int argc, const char* argv[])
{
    test::run(argc, argv);
    return 0;
}
