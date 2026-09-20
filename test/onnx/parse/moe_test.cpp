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

#include <onnx_test.hpp>
#include <migraphx/op/builder/insert.hpp>

TEST_CASE(moe_test)
{
    EXPECT(
        check_parse("moe_test.onnx",
                    {{"input", {migraphx::shape::float_type, {1, 2, 4}}},
                     {"router_probs", {migraphx::shape::float_type, {2, 3}}},
                     {"fc1_experts_weights", {migraphx::shape::float_type, {3, 2, 4}}},
                     {"fc1_experts_bias", {migraphx::shape::float_type, {3, 2}}},
                     {"fc2_experts_weights", {migraphx::shape::float_type, {3, 4, 2}}},
                     {"fc2_experts_bias", {migraphx::shape::float_type, {3, 4}}}},
                    [](migraphx::module& m, const auto& args) {
                        auto u = m.add_instruction(migraphx::make_op("undefined"));
                        auto r = migraphx::op::builder::add(
                            "moe",
                            m,
                            {args[0], args[1], args[2], u, args[3], args[4], u, args[5], u, u, u},
                            {{"activation_type", "relu"},
                             {"activation_alpha", 1.0f},
                             {"activation_beta", 0.0f},
                             {"k", 2},
                             {"normalize_routing_weights", true},
                             {"swiglu_fusion", 0}});
                        m.add_return({r.at(0)});
                    }));
}

TEST_CASE(moe_gated_test)
{
    EXPECT(
        check_parse("moe_gated_test.onnx",
                    {{"input", {migraphx::shape::float_type, {1, 2, 4}}},
                     {"router_probs", {migraphx::shape::float_type, {2, 3}}},
                     {"fc1_experts_weights", {migraphx::shape::float_type, {3, 2, 4}}},
                     {"fc2_experts_weights", {migraphx::shape::float_type, {3, 4, 2}}},
                     {"fc3_experts_weights", {migraphx::shape::float_type, {3, 2, 4}}}},
                    [](migraphx::module& m, const auto& args) {
                        // The empty optional bias inputs share one undefined instruction and
                        // the parser adds another one for the unset scale and bias slots
                        auto ug = m.add_instruction(migraphx::make_op("undefined"));
                        auto up = m.add_instruction(migraphx::make_op("undefined"));
                        auto r  = migraphx::op::builder::add(
                            "moe",
                            m,
                            {args[0], args[1], args[2], up, ug, args[3], up, ug, args[4], up, up},
                            {{"activation_type", "silu"},
                             {"activation_alpha", 1.0f},
                             {"activation_beta", 0.0f},
                             {"k", 2},
                             {"normalize_routing_weights", true},
                             {"swiglu_fusion", 0}});
                        m.add_return({r.at(0)});
                    }));
}

TEST_CASE(moe_swiglu_test)
{
    EXPECT(
        check_parse("moe_swiglu_test.onnx",
                    {{"input", {migraphx::shape::float_type, {1, 2, 4}}},
                     {"router_probs", {migraphx::shape::float_type, {2, 3}}},
                     {"fc1_experts_weights", {migraphx::shape::float_type, {3, 4, 4}}},
                     {"fc1_experts_bias", {migraphx::shape::float_type, {3, 4}}},
                     {"fc2_experts_weights", {migraphx::shape::float_type, {3, 4, 2}}},
                     {"fc2_experts_bias", {migraphx::shape::float_type, {3, 4}}}},
                    [](migraphx::module& m, const auto& args) {
                        auto u = m.add_instruction(migraphx::make_op("undefined"));
                        auto r = migraphx::op::builder::add(
                            "moe",
                            m,
                            {args[0], args[1], args[2], u, args[3], args[4], u, args[5], u, u, u},
                            {{"activation_type", "swiglu"},
                             {"activation_alpha", 1.702f},
                             {"activation_beta", 1.0f},
                             {"k", 2},
                             {"normalize_routing_weights", true},
                             {"swiglu_fusion", 1},
                             {"swiglu_limit", 7.0f}});
                        m.add_return({r.at(0)});
                    }));
}

TEST_CASE(moe_sparse_mixer_test)
{
    EXPECT(test::throws([&] { optimize_onnx("moe_sparse_mixer_test.onnx"); }));
}
