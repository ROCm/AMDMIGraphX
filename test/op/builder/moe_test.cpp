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

#include <op_builder_test_utils.hpp>

#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

// Expected values are computed with a float64 numpy implementation of the
// com.microsoft MoE/QMoE specification, cross-checked against onnxruntime

namespace {

struct moe_runner
{
    migraphx::module mm;
    migraphx::parameter_map params;

    migraphx::instruction_ref
    add_input(const std::string& name, const migraphx::shape& s, const std::vector<float>& data)
    {
        params[name] = migraphx::literal{s, data}.get_argument();
        return mm.add_parameter(name, s);
    }

    migraphx::instruction_ref add_quant_input(const std::string& name,
                                              const migraphx::shape& s,
                                              const std::vector<uint8_t>& data)
    {
        params[name] = migraphx::literal{s, data}.get_argument();
        return mm.add_parameter(name, s);
    }

    migraphx::instruction_ref undefined()
    {
        return mm.add_instruction(migraphx::make_op("undefined"));
    }

    std::vector<float> run(const std::vector<migraphx::instruction_ref>& args,
                           const migraphx::value& options)
    {
        migraphx::op::builder::add("moe", mm, args, options);
        migraphx::program p{std::move(mm)};
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval(params).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
        return result_vector;
    }
};

} // namespace

TEST_CASE(moe_relu_top2_normalized_test)
{
    // 2 tokens, hidden=2, 3 experts, inter=2, k=2 with renormalized routing
    moe_runner r;
    auto x = r.add_input("x", {migraphx::shape::float_type, {2, 2}}, {1.0f, 2.0f, -1.0f, 0.5f});
    auto router = r.add_input(
        "router", {migraphx::shape::float_type, {2, 3}}, {2.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f});
    auto w1 =
        r.add_input("w1",
                    {migraphx::shape::float_type, {3, 2, 2}},
                    {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f});
    auto b1 = r.add_input(
        "b1", {migraphx::shape::float_type, {3, 2}}, {0.5f, 0.0f, 0.0f, 0.5f, -0.5f, 0.0f});
    auto w2 = r.add_input("w2",
                          {migraphx::shape::float_type, {3, 2, 2}},
                          {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    auto b2 = r.add_input(
        "b2", {migraphx::shape::float_type, {3, 2}}, {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f});
    auto u = r.undefined();

    auto result =
        r.run({x, router, w1, u, b1, w2, u, b2},
              {{"activation_type", "relu"}, {"k", 2}, {"normalize_routing_weights", true}});
    std::vector<float> expected = {3.36552929f, 2.73105858f, 0.268941421f, 0.0f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(moe_swiglu_interleaved_test)
{
    // Fused interleaved swiglu with clamping limit, alpha and beta
    moe_runner r;
    auto x      = r.add_input("x", {migraphx::shape::float_type, {1, 2}}, {1.0f, -2.0f});
    auto router = r.add_input("router", {migraphx::shape::float_type, {1, 2}}, {1.0f, 0.0f});
    auto w1     = r.add_input("w1",
                              {migraphx::shape::float_type, {2, 4, 2}},
                              {1.0f,
                               0.0f,
                               0.0f,
                               1.0f,
                               1.0f,
                               1.0f,
                               0.0f,
                               -1.0f,
                               -1.0f,
                               0.0f,
                               1.0f,
                               0.0f,
                               0.0f,
                               1.0f,
                               1.0f,
                               1.0f});
    auto b1     = r.add_input("b1",
                              {migraphx::shape::float_type, {2, 4}},
                              {0.1f, 0.2f, -0.1f, 0.0f, 0.0f, 0.1f, 0.2f, -0.2f});
    auto w2     = r.add_input("w2",
                              {migraphx::shape::float_type, {2, 2, 2}},
                              {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.0f, 1.0f});
    auto b2 = r.add_input("b2", {migraphx::shape::float_type, {2, 2}}, {0.3f, 0.0f, 0.0f, -0.3f});
    auto u  = r.undefined();

    auto result                 = r.run({x, router, w1, u, b1, w2, u, b2},
                                        {{"activation_type", "swiglu"},
                                         {"swiglu_fusion", 1},
                                         {"activation_alpha", 1.702f},
                                         {"activation_beta", 1.0f},
                                         {"swiglu_limit", 0.5f},
                                         {"k", 2},
                                         {"normalize_routing_weights", true}});
    std::vector<float> expected = {0.295990475f, -0.252263394f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(moe_silu_gated_fc3_test)
{
    // Mixtral style: inter = silu(fc1(x)) * fc3(x), top-1 routing, no biases
    moe_runner r;
    auto x = r.add_input("x", {migraphx::shape::float_type, {2, 2}}, {1.0f, 2.0f, 3.0f, -1.0f});
    auto router =
        r.add_input("router", {migraphx::shape::float_type, {2, 2}}, {3.0f, 0.0f, 0.0f, 3.0f});
    auto w1 = r.add_input("w1",
                          {migraphx::shape::float_type, {2, 2, 2}},
                          {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    auto w2 = r.add_input("w2",
                          {migraphx::shape::float_type, {2, 2, 2}},
                          {1.0f, 1.0f, 1.0f, -1.0f, 2.0f, 0.0f, 0.0f, 2.0f});
    auto w3 = r.add_input("w3",
                          {migraphx::shape::float_type, {2, 2, 2}},
                          {1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 2.0f, 1.0f, 1.0f});
    auto u  = r.undefined();

    auto result =
        r.run({x, router, w1, u, u, w2, u, u, w3}, {{"activation_type", "silu"}, {"k", 1}});
    std::vector<float> expected = {0.411113447f, 3.76721148f, 1.02474656f, 10.8887696f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(moe_int4_blockwise_zero_points_test)
{
    // 4-bit weights packed along in_features, blockwise scales (block=2) and
    // packed blockwise zero points, fused interleaved swiglu
    moe_runner r;
    auto x = r.add_input("x", {migraphx::shape::float_type, {1, 4}}, {1.0f, -1.0f, 2.0f, 0.5f});
    auto router = r.add_input("router", {migraphx::shape::float_type, {1, 2}}, {1.0f, -1.0f});
    auto w1 =
        r.add_quant_input("w1",
                          {migraphx::shape::uint8_type, {2, 4, 2}},
                          {195, 23, 15, 148, 82, 139, 230, 161, 136, 240, 113, 45, 73, 54, 188, 5});
    auto s1 = r.add_input("s1",
                          {migraphx::shape::float_type, {2, 4, 2}},
                          {0.1f,
                           0.2f,
                           0.05f,
                           0.1f,
                           0.2f,
                           0.05f,
                           0.1f,
                           0.1f,
                           0.2f,
                           0.1f,
                           0.1f,
                           0.05f,
                           0.05f,
                           0.2f,
                           0.1f,
                           0.2f});
    auto z1 = r.add_quant_input(
        "z1", {migraphx::shape::uint8_type, {2, 4, 1}}, {135, 150, 120, 165, 136, 103, 89, 74});
    auto b1 = r.add_input("b1",
                          {migraphx::shape::float_type, {2, 4}},
                          {0.1f, -0.1f, 0.2f, 0.0f, 0.0f, 0.1f, -0.2f, 0.1f});
    auto w2 = r.add_quant_input(
        "w2", {migraphx::shape::uint8_type, {2, 4, 1}}, {180, 46, 150, 208, 90, 195, 24, 247});
    auto s2 = r.add_input("s2",
                          {migraphx::shape::float_type, {2, 4, 1}},
                          {0.1f, 0.2f, 0.1f, 0.05f, 0.2f, 0.1f, 0.05f, 0.1f});
    // Zero points for a single block per column: unpacking yields a padded
    // nibble which must be sliced away
    auto z2 = r.add_quant_input(
        "z2", {migraphx::shape::uint8_type, {2, 4, 1}}, {8, 7, 9, 6, 5, 8, 7, 10});
    auto b2 = r.add_input("b2",
                          {migraphx::shape::float_type, {2, 4}},
                          {0.1f, 0.0f, -0.1f, 0.2f, 0.0f, 0.2f, 0.1f, -0.1f});
    auto u  = r.undefined();

    auto result                 = r.run({x, router, w1, s1, b1, w2, s2, b2, u, u, u, z1, z2},
                                        {{"activation_type", "swiglu"},
                                         {"swiglu_fusion", 1},
                                         {"activation_alpha", 1.702f},
                                         {"activation_beta", 1.0f},
                                         {"swiglu_limit", 7.0f},
                                         {"k", 2},
                                         {"normalize_routing_weights", true},
                                         {"expert_weight_bits", 4}});
    std::vector<float> expected = {0.0817311461f, 0.00146826224f, -0.0711477744f, 0.186363335f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(moe_int8_per_column_scales_test)
{
    // 8-bit weights with 2D per-column scales and the default zero point of 128
    moe_runner r;
    auto x = r.add_input("x",
                         {migraphx::shape::float_type, {2, 4}},
                         {1.0f, 2.0f, -1.0f, 0.0f, 0.5f, -0.5f, 1.0f, 2.0f});
    auto router =
        r.add_input("router", {migraphx::shape::float_type, {2, 2}}, {0.0f, 2.0f, 2.0f, 0.0f});
    auto w1 = r.add_quant_input(
        "w1",
        {migraphx::shape::uint8_type, {2, 2, 4}},
        {130, 120, 140, 128, 125, 135, 128, 122, 128, 138, 118, 130, 132, 126, 128, 140});
    auto s1 = r.add_input("s1", {migraphx::shape::float_type, {2, 2}}, {0.1f, 0.2f, 0.05f, 0.1f});
    auto w2 = r.add_quant_input(
        "w2",
        {migraphx::shape::uint8_type, {2, 4, 2}},
        {120, 136, 128, 124, 132, 128, 126, 130, 140, 116, 128, 132, 124, 128, 130, 126});
    auto s2 = r.add_input("s2",
                          {migraphx::shape::float_type, {2, 4}},
                          {0.1f, 0.2f, 0.1f, 0.05f, 0.2f, 0.1f, 0.05f, 0.1f});
    auto u  = r.undefined();

    auto result = r.run({x, router, w1, s1, u, w2, s2},
                        {{"activation_type", "relu"}, {"k", 1}, {"expert_weight_bits", 8}});
    std::vector<float> expected = {3.17086948f,
                                   0.0f,
                                   -0.264239123f,
                                   0.264239123f,
                                   -1.19788403f,
                                   0.0f,
                                   0.598942013f,
                                   -0.149735503f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}
