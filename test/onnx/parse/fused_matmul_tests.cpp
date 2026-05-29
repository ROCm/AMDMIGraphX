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

// Core parse-level coverage for com.microsoft.FusedMatMul. Each test parses an .onnx
// fixture containing one FusedMatMul node and compares the parsed program against the
// expected lowering that parse_fused_matmul should produce.
//
// Attribute matrix covered here:
//   * no attributes set (2-D and batched 3-D)
//   * alpha           (scalar post-multiply on the product)
//   * transA / transB (swap the last two dims of A / B before the multiply)
//   * transBatchA / transBatchB
//       (for rank-R, permute [1, 2, ..., R-2, 0, R-1]; matches ORT's MatMulComputeHelper)
//   * a rank-error case where transBatch is set on rank-2 inputs (must throw)
//
// Additional parse-level scenarios (1-D promotion, batch broadcasting, dynamic shapes)
// live in fused_matmul_scenarios_tests.cpp.

#include <onnx_test.hpp>

// Plain 2-D * 2-D with no attributes. The parser should lower straight to a single `dot`.
TEST_CASE(fused_matmul_2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    mm->add_instruction(migraphx::make_op("dot"), l0, l1);

    auto prog = optimize_onnx("fused_matmul_2d_test.onnx");
    EXPECT(p == prog);
}

// alpha attribute. A non-1 alpha is applied as `mul(dot, multibroadcast(literal(alpha)))`
// on the product.
TEST_CASE(fused_matmul_alpha_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto dot = mm->add_instruction(migraphx::make_op("dot"), l0, l1);
    auto alpha_lit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.75f}});
    auto alpha_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), alpha_lit);
    mm->add_instruction(migraphx::make_op("mul"), dot, alpha_bc);

    auto prog = optimize_onnx("fused_matmul_alpha_test.onnx");
    EXPECT(p == prog);
}

// 3-D * 3-D with matching batch dim and no attributes. Batch dims already match, so the
// op-builder for `dot` inserts no multibroadcast and the parser lowers to a single `dot`.
TEST_CASE(fused_matmul_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3, 7, 8}});
    mm->add_instruction(migraphx::make_op("dot"), l0, l1);

    auto prog = optimize_onnx("fused_matmul_batch_test.onnx");
    EXPECT(p == prog);
}

// transA=1: A is transposed on its last two dims (perm [1, 0] for a rank-2 A) before the
// dot.
TEST_CASE(fused_matmul_trans_a_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 6}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto ta  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l0);
    mm->add_instruction(migraphx::make_op("dot"), ta, l1);

    auto prog = optimize_onnx("fused_matmul_trans_a_test.onnx");
    EXPECT(p == prog);
}

// transB=1: B is transposed on its last two dims before the dot.
TEST_CASE(fused_matmul_trans_b_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {8, 7}});
    auto tb  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    mm->add_instruction(migraphx::make_op("dot"), l0, tb);

    auto prog = optimize_onnx("fused_matmul_trans_b_test.onnx");
    EXPECT(p == prog);
}

// transA=1 AND transB=1: both inputs are transposed before the dot.
TEST_CASE(fused_matmul_trans_ab_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7, 6}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {8, 7}});
    auto ta  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l0);
    auto tb  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
    mm->add_instruction(migraphx::make_op("dot"), ta, tb);

    auto prog = optimize_onnx("fused_matmul_trans_ab_test.onnx");
    EXPECT(p == prog);
}

// transBatchA=1 on a rank-3 tensor. Permutation [1, 0, 2] moves dim-0 past the other
// batch dim(s) and leaves the matrix-col dim in place, matching ORT's MatMulComputeHelper
// for rank-3: [d0, d1, d2] -> [d1, d0, d2]. Applied before the (not-set) transA.
TEST_CASE(fused_matmul_trans_batch_a_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 3, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {3, 7, 8}});
    auto tba =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), l0);
    mm->add_instruction(migraphx::make_op("dot"), tba, l1);

    auto prog = optimize_onnx("fused_matmul_trans_batch_a_test.onnx");
    EXPECT(p == prog);
}

// transBatchB=1 on a rank-3 tensor. Same permutation [1, 0, 2], applied to B only.
TEST_CASE(fused_matmul_trans_batch_b_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 3, 8}});
    auto tbb =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), l1);
    mm->add_instruction(migraphx::make_op("dot"), l0, tbb);

    auto prog = optimize_onnx("fused_matmul_trans_batch_b_test.onnx");
    EXPECT(p == prog);
}

// Rank-error case: transBatchA is set on rank-2 inputs. ORT's MatMulComputeHelper
// requires both operands to have the same rank and rank >= 3 when any transBatch is set,
// so parsing must throw.
TEST_CASE(fused_matmul_trans_batch_rank_error_test)
{
    EXPECT(test::throws([&] { read_onnx("fused_matmul_trans_batch_rank_error_test.onnx"); }));
}
