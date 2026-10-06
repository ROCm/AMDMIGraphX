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

TEST_CASE(qmoe_test)
{
    EXPECT(check_parse("qmoe_test.onnx",
                       {{"input", {migraphx::shape::half_type, {1, 2, 4}}},
                        {"router_probs", {migraphx::shape::half_type, {2, 2}}},
                        {"fc1_experts_weights", {migraphx::shape::uint8_type, {2, 4, 2}}},
                        {"fc1_scales", {migraphx::shape::half_type, {2, 4, 2}}},
                        {"fc1_experts_bias", {migraphx::shape::half_type, {2, 4}}},
                        {"fc2_experts_weights", {migraphx::shape::uint8_type, {2, 4, 1}}},
                        {"fc2_scales", {migraphx::shape::half_type, {2, 4, 1}}},
                        {"fc2_experts_bias", {migraphx::shape::half_type, {2, 4}}},
                        {"fc1_zero_points", {migraphx::shape::uint8_type, {2, 4, 1}}},
                        {"fc2_zero_points", {migraphx::shape::uint8_type, {2, 4, 1}}}},
                       [](migraphx::module& m, const auto& args) {
                           // The empty fc3 inputs share one undefined instruction
                           auto u = m.add_instruction(migraphx::make_op("undefined"));
                           auto r = migraphx::op::builder::add("moe",
                                                               m,
                                                               {args[0],
                                                                args[1],
                                                                args[2],
                                                                args[3],
                                                                args[4],
                                                                args[5],
                                                                args[6],
                                                                args[7],
                                                                u,
                                                                u,
                                                                u,
                                                                args[8],
                                                                args[9]},
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

TEST_CASE(qmoe_int8_test)
{
    EXPECT(check_parse(
        "qmoe_int8_test.onnx",
        {{"input", {migraphx::shape::half_type, {1, 2, 4}}},
         {"router_probs", {migraphx::shape::half_type, {2, 2}}},
         {"fc1_experts_weights", {migraphx::shape::uint8_type, {2, 4, 4}}},
         {"fc1_scales", {migraphx::shape::half_type, {2, 4, 2}}},
         {"fc1_experts_bias", {migraphx::shape::half_type, {2, 4}}},
         {"fc2_experts_weights", {migraphx::shape::uint8_type, {2, 4, 2}}},
         {"fc2_scales", {migraphx::shape::half_type, {2, 4, 1}}},
         {"fc2_experts_bias", {migraphx::shape::half_type, {2, 4}}}},
        [](migraphx::module& m, const auto& args) {
            auto r = migraphx::op::builder::add(
                "moe",
                m,
                {args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]},
                {{"activation_type", "swiglu"},
                 {"activation_alpha", 1.702f},
                 {"activation_beta", 1.0f},
                 {"k", 2},
                 {"normalize_routing_weights", true},
                 {"swiglu_fusion", 1},
                 {"swiglu_limit", 7.0f},
                 {"expert_weight_bits", 8}});
            m.add_return({r.at(0)});
        }));
}

TEST_CASE(qmoe_fp4_test)
{
    EXPECT(test::throws([&] { optimize_onnx("qmoe_fp4_test.onnx"); }));
}
