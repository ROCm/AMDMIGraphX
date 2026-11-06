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

#include <migraphx/op/builder/insert.hpp>
#include <onnx_test.hpp>

namespace {

migraphx::instruction_ref lower_matmul_nbits(migraphx::module& m,
                                             const std::vector<migraphx::instruction_ref>& args)
{
    migraphx::value options;
    options.insert({"N", 16});
    options.insert({"K", 16});
    options.insert({"bits", 4});
    options.insert({"block_size", 16});
    return migraphx::op::builder::add("matmulnbits", m, args, options).at(0);
}

std::pair<migraphx::instruction_ref, migraphx::instruction_ref>
skip_simplified_ln(migraphx::module& m,
                   migraphx::instruction_ref x,
                   migraphx::instruction_ref skip,
                   migraphx::instruction_ref gamma)
{
    auto x_dtype  = x->get_shape().type();
    int64_t axis  = static_cast<int64_t>(x->get_shape().ndim()) - 1;
    auto skip_sum = add_common_op(m, migraphx::make_op("add"), {x, skip});
    auto float_x  = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), skip_sum);
    auto x_sq = add_common_op(m, migraphx::make_op("mul"), {float_x, float_x});
    auto rms  = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {axis}}}), x_sq);
    auto eps =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5f}});
    rms       = add_common_op(m, migraphx::make_op("add"), {rms, eps});
    auto rrms = m.add_instruction(migraphx::make_op("rsqrt"), rms);
    if(rrms->get_shape().type() != x_dtype)
    {
        rrms = m.add_instruction(migraphx::make_op("convert", {{"target_type", x_dtype}}), rrms);
    }
    auto result = add_common_op(m, migraphx::make_op("mul"), {skip_sum, rrms});
    result      = add_common_op(m, migraphx::make_op("mul"), {result, gamma});
    return {result, skip_sum};
}

} // namespace

TEST_CASE(ssmlp_fp16_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto half     = migraphx::shape::half_type;
    auto residual = mm->add_parameter("residual", migraphx::shape{half, {1, 16}});
    auto attn     = mm->add_parameter("attn", migraphx::shape{half, {1, 16}});
    auto gamma1   = mm->add_parameter("gamma1", migraphx::shape{half, {16}});
    auto gate_b =
        mm->add_parameter("gate_b", migraphx::shape{migraphx::shape::uint8_type, {16, 1, 8}});
    auto gate_s = mm->add_parameter("gate_s", migraphx::shape{half, {16}});
    auto gate_z = mm->add_parameter("gate_z", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto up_b = mm->add_parameter("up_b", migraphx::shape{migraphx::shape::uint8_type, {16, 1, 8}});
    auto up_s = mm->add_parameter("up_s", migraphx::shape{half, {16}});
    auto up_z = mm->add_parameter("up_z", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto down_b =
        mm->add_parameter("down_b", migraphx::shape{migraphx::shape::uint8_type, {16, 1, 8}});
    auto down_s = mm->add_parameter("down_s", migraphx::shape{half, {16}});
    auto down_z = mm->add_parameter("down_z", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto gamma2 = mm->add_parameter("gamma2", migraphx::shape{half, {16}});

    auto [mlp_in, residual_sum] = skip_simplified_ln(*mm, attn, residual, gamma1);
    auto gate                   = lower_matmul_nbits(*mm, {mlp_in, gate_b, gate_s, gate_z});
    auto up                     = lower_matmul_nbits(*mm, {mlp_in, up_b, up_s, up_z});
    auto sig                    = mm->add_instruction(migraphx::make_op("sigmoid"), gate);
    auto silu                   = add_common_op(*mm, migraphx::make_op("mul"), {gate, sig});
    auto mid                    = add_common_op(*mm, migraphx::make_op("mul"), {silu, up});
    auto down                   = lower_matmul_nbits(*mm, {mid, down_b, down_s, down_z});
    skip_simplified_ln(*mm, down, residual_sum, gamma2);

    auto prog = optimize_onnx("ssmlp_fp16_test.onnx");
    p.sort();
    prog.sort();
    EXPECT(p == prog);
}
