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

// Additional parse-level coverage for com.microsoft.FusedMatMul:
//   * numpy.matmul 1-D promotion (vector * matrix, matrix * vector, vector * vector,
//     batched-matrix * vector)
//   * batch-dim broadcasting with a rank mismatch, with and without the alpha attribute
//   * dynamic input shapes (which the parser previously rejected outright)
//
// Each test builds the program MIGraphX is expected to produce and compares it against
// the program parsed from a generated .onnx fixture. The expected lowering for a static-
// shape, non-transposed FusedMatMul is:
//
//   [optional unsqueeze] -> [optional multibroadcast] -> dot -> [optional mul(alpha)]
//   -> [optional squeeze to undo 1-D promotion]
//
// so each test just emits that pattern directly.

#include <onnx_test.hpp>

// numpy.matmul 1-D promotion on A: a `1` is prepended to A's shape for the multiply and
// squeezed off the result afterward.
//   [7] * [7, 8] -> unsqueeze axis 0 -> [1, 7] * [7, 8] -> dot -> [1, 8] -> squeeze axis 0 -> [8]
TEST_CASE(fused_matmul_vm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7, 8}});
    auto u0  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), u0, l1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), dot);

    auto prog = optimize_onnx("fused_matmul_vm_test.onnx");
    EXPECT(p == prog);
}

// numpy.matmul 1-D promotion on B: a `1` is appended to B's shape for the multiply and
// squeezed off the result afterward.
//   [6, 7] * [7] -> [6, 7] * unsqueeze axis 1 [7, 1] -> dot -> [6, 1] -> squeeze axis 1 -> [6]
TEST_CASE(fused_matmul_mv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto u1  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), l0, u1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), dot);

    auto prog = optimize_onnx("fused_matmul_mv_test.onnx");
    EXPECT(p == prog);
}

// Both inputs 1-D: both promotions apply and both are undone afterward.
//   [7] * [7] -> unsqueeze axis 0 on A + unsqueeze axis 1 on B -> [1, 7] * [7, 1]
//   -> dot -> [1, 1] -> squeeze axis 0 -> [1] -> squeeze axis 0 -> scalar
// The final double squeeze matches numpy.matmul's "1-D * 1-D returns a scalar inner
// product" semantics.
TEST_CASE(fused_matmul_vv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto u0  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto u1  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), u0, u1);
    auto sr0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), dot);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sr0);

    auto prog = optimize_onnx("fused_matmul_vv_test.onnx");
    EXPECT(p == prog);
}

// Batched-matrix times 1-D vector: combines 1-D promotion on B with batch broadcasting.
//   [3, 6, 7] * [7] -> unsqueeze axis 1 on B -> [3, 6, 7] * [7, 1]
//   -> multibroadcast B to [3, 7, 1] (op-builder for `dot`)
//   -> dot -> [3, 6, 1] -> squeeze axis 2 -> [3, 6]
TEST_CASE(fused_matmul_bmv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 6, 7}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {7}});
    auto u1  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto bu1 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 7, 1}}}), u1);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), l0, bu1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), dot);

    auto prog = optimize_onnx("fused_matmul_bmv_test.onnx");
    EXPECT(p == prog);
}

// Batch broadcasting with a rank mismatch between the two operands. The op-builder for
// `dot` inserts a multibroadcast on whichever side needs the extra batch dims so that
// the underlying dot sees matching leading dims.
//   [2, 3, 4] * [4, 5] -> multibroadcast B to [2, 4, 5] -> dot -> [2, 3, 5]
TEST_CASE(fused_matmul_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto bl1 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("dot"), l0, bl1);

    auto prog = optimize_onnx("fused_matmul_bcast_test.onnx");
    EXPECT(p == prog);
}

// Batch broadcasting combined with the alpha attribute. A has leading dim 1, B has
// leading dim 2, so A is multibroadcast to [2, 3, 4]. Afterwards the result is scaled
// by alpha=0.25, which the parser lowers to a literal + multibroadcast + mul.
//   [1, 3, 4] * [2, 4, 5] -> multibroadcast A to [2, 3, 4] -> dot -> [2, 3, 5]
//                         -> mul by broadcast(alpha_lit)
TEST_CASE(fused_matmul_bcast_alpha_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 3, 4}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2, 4, 5}});
    auto bl0 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), l0);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), bl0, l1);
    auto alpha_lit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.25f}});
    auto alpha_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", dot->get_shape().lens()}}), alpha_lit);
    mm->add_instruction(migraphx::make_op("mul"), dot, alpha_bc);

    auto prog = optimize_onnx("fused_matmul_bcast_alpha_test.onnx");
    EXPECT(p == prog);
}

// Dynamic input shapes must parse without throwing. With both operands rank 2 and their
// (zero) batch dims trivially matching, the dynamic path of the op-builder for `dot`
// emits a plain dot with no broadcast_for_dot inserted. Guards against regressing the
// removal of the `dynamic inputs not supported` throw in parse_fused_matmul.
TEST_CASE(fused_matmul_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {{4, 8, {6}}, {7, 7}}});
    auto l1 =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {{7, 7}, {1, 5, {3}}}});
    auto ret = mm->add_instruction(migraphx::make_op("dot"), l0, l1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["1"] = {{4, 8, {6}}, {7, 7}};
    options.map_dyn_input_dims["2"] = {{7, 7}, {1, 5, {3}}};
    auto prog                       = read_onnx("fused_matmul_dyn_test.onnx", options);

    EXPECT(p == prog);
}
