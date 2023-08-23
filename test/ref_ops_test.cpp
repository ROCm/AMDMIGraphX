/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <migraphx/literal.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"
#include <migraphx/half.hpp>
#include <iomanip>

float sigmoid(float x) { return 1 / (1 + expf(-x)); }

float elu(float a, float x) { return x > 0 ? x : a * std::expm1(x); }

TEST_CASE(abs_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = mm->add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    mm->add_instruction(migraphx::make_op("abs"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(abs_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 8}, {2, 2}}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("abs"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {-1, 2, -3, 4};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 2}};
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(acos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-0.8f, 0.0f, 1.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("acos"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acosf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(acos_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("acos"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-0.8f, 0.0f, 1.0f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acosf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(acosh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{1.1f, 1.2f, 2.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("acosh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acoshf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(acosh_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    std::vector<float> input_data{1.1f, 1.2f, 2.0f};
    mm->add_instruction(migraphx::make_op("acosh"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return acoshf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(add_broadcast_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2}};
        std::vector<float> b_data{0, -1, -2, -3};
        uint64_t axis = 0;
        auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
        auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
        auto l3       = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}),
            l2);
        mm->add_instruction(migraphx::make_op("add"), l1, l3);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
        std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 1}};
        std::vector<float> b_data{0, -1, -2, -3};
        auto l1 = mm->add_literal(migraphx::literal{a_shape, a_data});
        auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
        auto l3 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l1);
        auto l4 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l2);
        mm->add_instruction(migraphx::make_op("add"), l3, l4);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape().packed());
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("add"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(add_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("add"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 1};
    std::vector<float> y_data{1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(argmax_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 0}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 2, 1, 2, 0, 0, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {1, 3, 2, 2, 2, 3};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_test_neg_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {0, 0, 2, 1, 2, 0, 0, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", -2}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmax_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {3, 6}, {3, 6}}};
    auto dl = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("argmax", {{"axis", 0}}), dl);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 3, 4}};
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<int64_t> res_gold = {0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_0)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 2, 0, 2, 0, 1, 2, 0};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_2)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(argmin_test_neg_1)
{
    migraphx::program p;
    auto* mm                = p.get_main_module();
    std::vector<float> data = {1.2255,  1.6834,  -2.0305, -0.3221, 0.4701,  0.2583, 0.7545, 2.5758,
                               -1.6849, 0.0928,  0.9022,  -0.8765, -0.4090, 0.9301, 2.0724, -1.5706,
                               0.4867,  -0.1493, 0.6957,  -0.2179, 0.7142,  0.7177, 0.0183, 1.3497};
    std::vector<int64_t> res_gold = {2, 1, 0, 3, 3, 2};
    migraphx::shape data_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto dl = mm->add_literal(migraphx::literal{data_shape, data});
    mm->add_instruction(migraphx::make_op("argmin", {{"axis", -1}}), dl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(result_vec, res_gold));
}

TEST_CASE(asin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("asin"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(asin_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("asin"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-0.5f, 0.0f, 0.9f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(asinh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("asinh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(asinh_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("asinh"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-0.5f, 0.0f, 0.9f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return asinhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(atan_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-1.0f, 0.0f, 1.0f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("atan"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(atan_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("atan"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1.0f, 0.0f, 1.0f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(atanh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{0.4435683f, 0.6223626f, 0.316958f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("atanh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(atanh_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("atanh"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{0.4435683f, 0.6223626f, 0.316958f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return atanhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths = {2};
    op.padding = {0};
    op.stride  = {1};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.3, 0.25, 0.65, 0.7, 0.5, 0.4, 0.4, 0.35};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.3, 0.25, 0.65, 0.7, 0.5, 0.4, 0.4, 0.35};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_pad_test)
{
    // pooling with dynamic input and padding, ceiling mode for output size
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 3}, {2, 4}, {2, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2, 2}},
                                           {"padding", {1, 0}},
                                           {"ceil_mode", true},
                                           {"stride", {2, 2}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4, 5, 6};

    //      *  *  *
    //      1  2  3        padding will look like this
    //      4  5  6        The * are used when tiling the kernel
    //      *  *  *        but are ignored in averaging

    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 2, 3}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5, 3.0, 4.5, 6.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_pad_ceil_test)
{
    // pooling with dynamic input and padding
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 3}, {2, 4}, {2, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2, 3}},
                                           {"padding", {1, 2}},
                                           {"ceil_mode", true},
                                           {"stride", {1, 1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4};

    //  * *  *  * * *
    //  * *  1  2 * *      padded input will look like this
    //  * *  3  4 * *      but the * are ignored in averaging
    //  * *  *  * * *

    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // clang-format off
    std::vector<float> gold{1.0, 1.5, 1.5, 2.0, 
                            2.0, 2.5, 2.5, 3.0, 
                            3.0, 3.5, 3.5, 4.0};
    // clang-format on
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_stride2_test)
{
    // 1D case 2, stride 2
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 4}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths = {2};
    op.padding = {1};
    op.stride  = {2};

    // clang-format off
    std::vector<float> data{1.6321, -2.4186, 0.2239, -1.4232, 
                            0.8158, 0.4103, -0.3149, -0.1361,
                            -0.3442, 2.007, 0.4331, 1.5295,
                            0.9965, 0.4766, 1.0942, -0.2915};
    // clang-format on
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold{1.6321, -1.09735, -1.4232,
                            0.8158, 0.0477, -0.1361, 
                            -0.3442, 1.22005, 1.5295,
                            0.9965, 0.7854, -0.2915};
    // clang-format on
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(avgpool_rank5_test)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths = {2, 2, 2};
    op.padding = {0, 0, 0};
    op.stride  = {1, 1, 1};

    std::vector<float> data{
        -0.179, -1.756, 0.651,  1.955,  1.87,   -0.604, 0.247,  0.449,  -0.137, 1.187,  1.593,
        0.424,  2.698,  -0.104, -0.069, -1.293, 0.538,  1.291,  0.974,  1.096,  0.74,   -0.669,
        -1.08,  -1.041, -1.407, 1.43,   -0.211, -0.017, 0.532,  1.276,  0.627,  0.236,  -0.396,
        -0.204, 0.501,  -0.599, -1.414, -0.615, -0.274, 0.168,  -0.144, 0.5,    1.42,   1.082,
        -0.952, -0.846, -1.244, 1.475,  1.246,  1.344,  -1.722, -1.24,  -0.851, 0.06,   0.507,
        0.762,  -0.007, -1.484, 1.028,  0.317,  1.077,  -1.289, 0.875,  -0.417, -0.673, 1.715,
        -0.307, 0.264,  -0.973, 1.412,  2.561,  -0.515, -0.201, 0.827,  -1.231, 1.958,  -0.552,
        0.036,  -0.993, -0.859, -1.458, -0.575, 0.048,  -0.779, -1.025, -1.135, 1.166,  -0.131,
        0.726,  0.52,   0.467,  -0.494, 0.675,  0.203,  -0.63,  -0.918, -0.5,   -1.395, 1.39,
        1.705,  0.444,  -0.835, -0.506, 0.101,  0.602,  0.543,  0.357,  1.042};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        0.908,     0.250625,  0.795,     0.40425, 0.711875,  0.194875,  0.014125,  0.09425,
        -0.078375, 0.139375,  0.46075,   0.0285,  -0.188125, -0.085,    0.378125,  -0.085375,
        -0.04,     0.304125,  0.40775,   0.2835,  0.112375,  -0.073375, 0.4355,    -0.187,
        -0.392625, -0.258375, -0.485875, -0.0345, 0.16125,   -0.131875, -0.228375, 0.068625};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}), l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(broadcast_2in_static_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("broadcast", {{"axis", axis}}), l2, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(broadcast_2in_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {{2, 2}, {2, 4}}};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    uint64_t axis = 0;
    auto pa       = mm->add_parameter("a", a_shape);
    auto lb       = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("broadcast", {{"axis", axis}}), lb, pa);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape input_fixed_shape0{migraphx::shape::int32_type, {2, 2}};
    migraphx::parameter_map params0;
    params0["a"] = migraphx::argument(input_fixed_shape0, a_data.data());
    auto result  = p.eval(params0).back();
    auto output  = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -2);
    EXPECT(output(1, 0) == -3);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(ceil_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data = {1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("ceil"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::ceil(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(ceil_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{4, 12};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("ceil"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::ceil(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l       = mm->add_literal(migraphx::literal{s, {-1.0, 0.0, 10.0}});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l, min_val, max_val);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(clip_dyn_test)
{
    migraphx::program p;
    auto* mm                                            = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dds = {{2, 8, {3}}};
    migraphx::shape s{migraphx::shape::float_type, dds};
    auto l       = mm->add_parameter("X", s);
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val      = mm->add_instruction(migraphx::make_op("multibroadcast"), min_val, l);
    max_val      = mm->add_instruction(migraphx::make_op("multibroadcast"), max_val, l);
    mm->add_instruction(migraphx::make_op("clip"), l, min_val, max_val);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape{migraphx::shape::float_type, {3}};
    migraphx::parameter_map params;
    std::vector<float> data = {-1.0, 0.0, 10.0};
    params["X"]             = migraphx::argument(static_shape, data.data());
    auto result             = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(concat_test)
{
    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = 1;
        std::vector<int> data0 = {0, 1, 5, 6};
        std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
        std::vector<int> data2 = {10, 20};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::make_target("ref"));
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
        std::vector<int> results_vector(2 * 6);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(migraphx::verify::verify_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({2, 6})));
        EXPECT(migraphx::verify::verify_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({6, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = -1;
        std::vector<int> data0 = {0, 1, 5, 6};
        std::vector<int> data1 = {2, 3, 4, 7, 8, 9};
        std::vector<int> data2 = {10, 20};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::int32_type, {2, 1}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::make_target("ref"));
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9, 20};
        std::vector<int> results_vector(2 * 6);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(migraphx::verify::verify_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({2, 6})));
        EXPECT(migraphx::verify::verify_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({6, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = 0;
        std::vector<int> data0 = {0, 1, 2, 3};
        std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
        std::vector<int> data2 = {10, 11};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::make_target("ref"));
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> results_vector(6 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(migraphx::verify::verify_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({6, 2})));
        EXPECT(migraphx::verify::verify_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({2, 1})));
    }

    {
        migraphx::program p;
        auto* mm               = p.get_main_module();
        int axis               = -2;
        std::vector<int> data0 = {0, 1, 2, 3};
        std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
        std::vector<int> data2 = {10, 11};
        migraphx::shape s0{migraphx::shape::int32_type, {2, 2}};
        migraphx::shape s1{migraphx::shape::int32_type, {3, 2}};
        migraphx::shape s2{migraphx::shape::int32_type, {1, 2}};
        auto l0 = mm->add_literal(migraphx::literal{s0, data0});
        auto l1 = mm->add_literal(migraphx::literal{s1, data1});
        auto l2 = mm->add_literal(migraphx::literal{s2, data2});
        mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), l0, l1, l2);
        p.compile(migraphx::make_target("ref"));
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> results_vector(6 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(migraphx::verify::verify_range(result.get_shape().lens(),
                                              std::vector<std::size_t>({6, 2})));
        EXPECT(migraphx::verify::verify_range(result.get_shape().strides(),
                                              std::vector<std::size_t>({2, 1})));
    }
}

TEST_CASE(concat_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    int axis = 0;
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2}}, {2, 3, {2}}}};
    migraphx::shape s1{migraphx::shape::int32_type, {{3, 4, {4}}, {2, 3, {2}}}};
    migraphx::shape s2{migraphx::shape::int32_type, {{1, 5, {3}}, {2, 3, {2}}}};

    auto input0 = mm->add_parameter("X", s0);
    auto input1 = mm->add_parameter("Y", s1);
    auto input2 = mm->add_parameter("Z", s2);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", axis}}), input0, input1, input2);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape0{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape static_shape1{migraphx::shape::int32_type, {3, 2}};
    migraphx::shape static_shape2{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> data0 = {0, 1, 2, 3};
    std::vector<int> data1 = {4, 5, 6, 7, 8, 9};
    std::vector<int> data2 = {10, 11};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(static_shape0, data0.data());
    params["Y"] = migraphx::argument(static_shape1, data1.data());
    params["Z"] = migraphx::argument(static_shape2, data2.data());
    auto result = p.eval(params).back();

    std::vector<int> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
    EXPECT(migraphx::verify::verify_range(result.get_shape().lens(),
                                          std::vector<std::size_t>({6, 2})));
}

TEST_CASE(contiguous_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(migraphx::literal{a_shape, data});
    mm->add_instruction(migraphx::make_op("contiguous"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(contiguous_param_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    auto a = mm->add_parameter("X", a_shape);
    mm->add_instruction(migraphx::make_op("contiguous"), a);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(a_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(contiguous_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape dyn_shape{migraphx::shape::float_type, {{1, 1}, {2, 6}, {2, 2}, {2, 2}}};
    auto input = mm->add_parameter("X", dyn_shape);
    mm->add_instruction(migraphx::make_op("contiguous"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape{migraphx::shape::float_type, {1, 3, 2, 2}, {12, 1, 6, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(static_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_strides = {12, 4, 2, 1};
    EXPECT(result.get_shape().strides() == new_strides);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(conv_dyn_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type,
                                    {{1, 100}, {3, 3}, {4, 4}, {4, 4}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {2, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    std::vector<float> sol = {-0.20817225,
                              0.87965256,
                              0.14958936,
                              -1.24887264,
                              -0.06540672,
                              0.20778663,
                              0.40456355,
                              -0.99900877,
                              0.4917807,
                              0.1994698,
                              0.64205718,
                              0.37798831,
                              -0.25315839,
                              0.44276932,
                              -0.16138598,
                              0.79344082};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv_dyn_img_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {4, 6}, {4, 6}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {1, 1}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.28007596, 0.46114671, 0.12171969, 0.52260835, 0.40916841, 0.07163955,
                            0.09896668, 0.98628836, 0.69406788, 0.44868846, 0.64017681, 0.27048886,
                            0.30187397, 0.07334207, 0.05258557, 0.80747513, 0.81330534, 0.00497161,
                            0.33005534, 0.08908686, 0.46794691, 0.61768946, 0.55104806, 0.13406187,
                            0.70244284, 0.61296941, 0.46742536, 0.29712714, 0.91839388, 0.0834397,
                            0.14476327, 0.37857075, 0.25922384, 0.61620963, 0.69455439, 0.70389431,
                            0.77388606, 0.1752363,  0.74631394, 0.24604889, 0.53600244, 0.22116457,
                            0.81217463, 0.10789447, 0.43083784, 0.63371852, 0.69742316, 0.09536905};

    std::vector<float> c = {0.98411968, 0.2899219,  0.44638833, 0.30390816, 0.03989896, 0.2445332,
                            0.32700131, 0.57517075, 0.06956476, 0.93079306, 0.19882314, 0.52940601,
                            0.35624753, 0.35938406, 0.9111428,  0.88923574, 0.61040283, 0.2797513,
                            0.15479768, 0.46534674, 0.16970931, 0.49704618, 0.07062198, 0.01678321,
                            0.53150934, 0.39244495, 0.9963813};

    std::vector<float> sol = {6.1329393, 4.3199925, 5.448438, 3.8497565};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(72);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, sol));

    a = {0.95600171, 0.20768181, 0.82844489, 0.14928212, 0.51280462, 0.1359196,  0.68903648,
         0.84174772, 0.425509,   0.956926,   0.82533291, 0.33821531, 0.57576055, 0.75330186,
         0.82710394, 0.93343847, 0.14499469, 0.74558021, 0.13935139, 0.90652876, 0.22611443,
         0.85323975, 0.30631787, 0.96983037, 0.51783421, 0.32247456, 0.28243352, 0.605865,
         0.33376446, 0.67864877, 0.15442507, 0.24977552, 0.86989425, 0.60036782, 0.26198306,
         0.1494149,  0.13678915, 0.24892094, 0.38282467, 0.64907906, 0.83756376, 0.77603195,
         0.33951558, 0.14856874, 0.45701939, 0.43786436, 0.57421759, 0.37326922, 0.63382506,
         0.11464436, 0.23309047, 0.76724102, 0.98712427, 0.80800108, 0.84296564, 0.79568268,
         0.45684131, 0.73867068, 0.57845499, 0.45073557, 0.27102442, 0.86460315, 0.06865567,
         0.81673446, 0.881835,   0.42351639, 0.83322931, 0.34101671, 0.51979151, 0.54920645,
         0.19287718, 0.33321689, 0.27752456, 0.45755893, 0.67484562, 0.68383122, 0.52361312,
         0.46437257, 0.50862936, 0.32460429, 0.1726007,  0.29933345, 0.64856728, 0.06471591,
         0.63370843, 0.27900152, 0.18595992, 0.48904812, 0.35368508, 0.09620202};

    c = {0.709561,   0.7916206,  0.0443115,  0.62592275, 0.2498623,  0.42725624, 0.7905135,
         0.53160169, 0.01303743, 0.01987505, 0.39041803, 0.89530203, 0.23155373, 0.44435213,
         0.14407301, 0.80968594, 0.38216188, 0.35692557, 0.2568538,  0.83587388, 0.43654904,
         0.04974508, 0.80375029, 0.25350374, 0.1820275,  0.23369029, 0.54358755};

    sol = {6.305986,
           5.564665,
           6.122996,
           5.7262855,
           5.5546584,
           5.779489,
           5.798161,
           5.160476,
           6.702436,
           5.4851074,
           6.227567,
           5.2016754};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 3, 6, 5}};

    migraphx::parameter_map params1;
    params1["X"] = migraphx::argument(input_fixed_shape1, a.data());
    params1["W"] = migraphx::argument(weights_shape, c.data());

    result = p.eval(params1).back();
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv_dyn_weights_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {0, 0}}, {"stride", {1, 1}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.28007596, 0.46114671, 0.12171969, 0.52260835, 0.40916841, 0.07163955,
                            0.09896668, 0.98628836, 0.69406788, 0.44868846, 0.64017681, 0.27048886,
                            0.30187397, 0.07334207, 0.05258557, 0.80747513, 0.81330534, 0.00497161,
                            0.33005534, 0.08908686, 0.46794691, 0.61768946, 0.55104806, 0.13406187,
                            0.70244284, 0.61296941, 0.46742536, 0.29712714, 0.91839388, 0.0834397,
                            0.14476327, 0.37857075, 0.25922384, 0.61620963, 0.69455439, 0.70389431,
                            0.77388606, 0.1752363,  0.74631394, 0.24604889, 0.53600244, 0.22116457,
                            0.81217463, 0.10789447, 0.43083784, 0.63371852, 0.69742316, 0.09536905};

    std::vector<float> c   = {0.98411968,
                            0.2899219,
                            0.44638833,
                            0.30390816,
                            0.03989896,
                            0.2445332,
                            0.32700131,
                            0.57517075,
                            0.06956476,
                            0.93079306,
                            0.19882314,
                            0.52940601};
    std::vector<float> sol = {1.9939406,
                              2.2703054,
                              1.8896171,
                              2.062202,
                              2.3035214,
                              1.629366,
                              2.1606991,
                              2.1917608,
                              1.6797699};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(72);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, sol));

    c   = {0.98411968, 0.2899219,  0.44638833, 0.30390816, 0.03989896, 0.2445332,  0.32700131,
         0.57517075, 0.06956476, 0.93079306, 0.19882314, 0.52940601, 0.35624753, 0.35938406,
         0.9111428,  0.88923574, 0.61040283, 0.2797513,  0.15479768, 0.46534674, 0.16970931,
         0.49704618, 0.07062198, 0.01678321, 0.53150934, 0.39244495, 0.9963813};
    sol = {6.1329393, 4.3199925, 5.448438, 3.8497565};
    migraphx::shape weights_fixed_shape1{migraphx::shape::float_type, {1, 3, 3, 3}};

    migraphx::parameter_map params1;
    params1["X"] = migraphx::argument(input_shape, a.data());
    params1["W"] = migraphx::argument(weights_fixed_shape1, c.data());

    result = p.eval(params1).back();
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv_dyn_img_same_upper_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_dyn_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {4, 6}, {4, 6}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};

    auto input   = mm->add_parameter("X", input_dyn_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};

    std::vector<float> c = {
        2.8150102e-01, 3.3198616e-01, 9.5149356e-01, 7.4039467e-02, 9.6555042e-01,
        2.8815505e-01, 2.5100240e-01, 5.2186239e-01, 2.3850012e-01,

        8.2963020e-01, 3.0763101e-04, 6.7026985e-01, 1.4260857e-01, 9.7517288e-01,
        3.6847427e-02, 8.5804445e-01, 7.3440993e-01, 6.7948365e-01,

        7.9253986e-02, 7.3943835e-01, 1.7813577e-01, 1.0780835e-01, 4.2304707e-01,
        4.0084350e-01, 1.1114500e-01, 4.4846520e-01, 5.0109702e-01};

    std::vector<float> sol = {3.013387,
                              3.7111127,
                              4.2946506,
                              3.579301,
                              4.5306826,
                              6.1262493,
                              6.332169,
                              4.495293,
                              4.46013,
                              6.0938954,
                              5.848162,
                              4.514299,
                              2.9587686,
                              4.117671,
                              3.5187216,
                              2.3236327};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 3, 4, 4}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    params0["W"] = migraphx::argument(weights_shape, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv_dyn_kernel_same_upper_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a   = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};
    std::vector<float> c   = {2.8150102e-01,
                            3.3198616e-01,
                            9.5149356e-01,
                            7.4039467e-02,

                            9.6555042e-01,
                            2.8815505e-01,
                            2.5100240e-01,
                            5.2186239e-01,

                            2.3850012e-01,
                            8.2963020e-01,
                            3.0763101e-04,
                            6.7026985e-01};
    std::vector<float> sol = {2.453681,
                              2.536207,
                              3.0187201,
                              1.7912633,
                              2.1738236,
                              2.9695358,
                              3.2319589,
                              1.859269,
                              2.5953722,
                              2.50734,
                              2.7736917,
                              1.2229807,
                              1.5900216,
                              0.9225286,
                              1.43048,
                              0.74341124};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv_dyn_kernel_same_lower_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 3}, {2, 3}}};

    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(
        migraphx::make_op(
            "convolution",
            {{"stride", {1, 1}}, {"padding_mode", migraphx::op::padding_mode_t::same_lower}}),
        input,
        weights);

    p.compile(migraphx::make_target("ref"));

    std::vector<float> a   = {0.63321185, 0.6466339,  0.8515352,  0.44240063, 0.5018913,  0.5068494,
                            0.75330657, 0.7383877,  0.15870683, 0.8171611,  0.56118083, 0.87004256,
                            0.24401724, 0.8815178,  0.4222333,  0.27191755,

                            0.41633207, 0.2460619,  0.32004243, 0.6962248,  0.12284133, 0.2620491,
                            0.96931046, 0.6030955,  0.7623861,  0.2395751,  0.61440414, 0.577285,
                            0.80087787, 0.12776066, 0.26566318, 0.46569306,

                            0.96701574, 0.3850145,  0.14165345, 0.5887347,  0.7152134,  0.5295342,
                            0.6303507,  0.4037548,  0.18556239, 0.79416305, 0.29107493, 0.18770285,
                            0.6870904,  0.30701008, 0.314684,   0.91075855};
    std::vector<float> c   = {2.8150102e-01,
                            3.3198616e-01,
                            9.5149356e-01,
                            7.4039467e-02,

                            9.6555042e-01,
                            2.8815505e-01,
                            2.5100240e-01,
                            5.2186239e-01,

                            2.3850012e-01,
                            8.2963020e-01,
                            3.0763101e-04,
                            6.7026985e-01};
    std::vector<float> sol = {0.91231215,
                              1.1416453,
                              1.00216,
                              1.6813052,
                              1.7131033,
                              2.453681,
                              2.536207,
                              3.0187201,
                              1.3293691,
                              2.1738236,
                              2.9695358,
                              3.2319589,
                              1.3228729,
                              2.5953722,
                              2.50734,
                              2.7736917};

    migraphx::shape weight_fixed_shape0{migraphx::shape::float_type, {1, 3, 2, 2}};

    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_shape, a.data());
    params0["W"] = migraphx::argument(weight_fixed_shape0, c.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, sol));
}

TEST_CASE(conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    std::vector<float> s = {-0.20817225,
                            0.87965256,
                            0.14958936,
                            -1.24887264,
                            -0.06540672,
                            0.20778663,
                            0.40456355,
                            -0.99900877,
                            0.4917807,
                            0.1994698,
                            0.64205718,
                            0.37798831,
                            -0.25315839,
                            0.44276932,
                            -0.16138598,
                            0.79344082};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(conv2d_padding_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.16115488, -0.09800646, -0.05412646, 0.10475694,  0.00555485,  -0.12667653, 0.0458357,
        -0.02656217, -0.16338061, 0.15037455,  0.0102711,   0.01303349,  0.05242859,  0.02034754,
        0.04751867,  -0.17038961, -0.1434752,  -0.10770349, 0.05676742,  -0.15838449, 0.10128359,
        -0.18958683, 0.11954515,  0.10758857,  -0.01058291, -0.12797487, 0.08971019,  0.18793164,
        -0.00881396, -0.06588994, -0.13321903, -0.03300409, 0.01439607,  0.07618178,  -0.11556662,
        0.00764295,  0.12956454,  -0.08937147, -0.12763587, 0.04674943,  0.05765297,  0.11336918,
        0.14747436,  -0.06199479, -0.01166052, -0.12432006, -0.04494537, -0.17581205, 0.09475745,
        0.1149437,   -0.1014564,  0.0274073,   -0.01323579, -0.11092556};

    std::vector<float> s = {
        -0.0201216,  0.40407312,  -0.39005592, -0.0631946,  0.37963012,  -0.64611685, 0.1349397,
        -0.54113752, 0.28533003,  0.27667275,  -0.16442731, -0.181494,   0.30564839,  0.58744538,
        0.32015014,  0.24969585,  -0.27367792, -0.53308117, 0.41236052,  0.26136363,  -0.01489828,
        0.57652152,  -0.38506854, 0.119615,    0.0437076,   0.04779706,  0.57887721,  0.23126155,
        0.05695833,  -0.68200272, 0.02063358,  -0.10267162, 0.8062973,   -0.38149622, -0.40134856,
        -0.03353126, 0.38991132,  -0.3478111,  0.03661491,  0.25783631,  0.62772679,  -0.1961118,
        0.76423508,  -0.36241418, -0.20994355, -0.12368261, -0.9406727,  0.02340185,  -0.08793129,
        -0.02471633, -0.58163726, -0.02211772, -0.42014724, 0.77525634,  0.504951,    -0.20537445,
        -0.20369984, -0.83037728, -1.40423918, -0.46160448, -0.22944322, 0.36074194,  0.49579027,
        0.46527559};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(conv2d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    std::vector<float> s = {0.27039781,
                            0.19105849,
                            -0.06339942,
                            -0.65087199,
                            0.40867025,
                            0.05063812,
                            -0.14907975,
                            0.49018705,
                            -0.49197209,
                            0.33236548,
                            -0.39374301,
                            0.16012701,
                            0.06574871,
                            0.71606487,
                            -0.55201721,
                            -0.46427044};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("convolution"), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(conv3d_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        2.82721668e-02,  6.44195229e-02,  1.53499246e-02,  1.72468081e-01,  -6.33238107e-02,
        9.49496776e-02,  1.40258059e-01,  -7.92879611e-02, -1.29301161e-01, 3.11307609e-03,
        -1.90624535e-01, 1.13238767e-01,  -2.80647576e-02, 3.12882811e-02,  -3.52091640e-02,
        3.33581865e-02,  6.43158704e-02,  7.40238279e-02,  -1.00106120e-01, -9.56912562e-02,
        1.44342467e-01,  9.40258950e-02,  6.36333972e-02,  1.66158378e-03,  -8.91554281e-02,
        2.58734226e-02,  1.70919895e-02,  1.78214177e-01,  8.84564668e-02,  8.98126513e-02,
        -1.63809001e-01, 1.37802169e-01,  1.66439757e-01,  -1.45631135e-02, 1.88469887e-04,
        4.76950556e-02,  -1.91969007e-01, -1.76233292e-01, -7.70473927e-02, 1.14828631e-01,
        1.76608220e-01,  -1.50728196e-01, 1.99946314e-02,  -5.88052124e-02, 1.31612435e-01,
        1.61106288e-02,  -1.35080189e-01, 1.49512306e-01,  3.86456847e-02,  1.29330024e-01,
        -3.22975963e-02, -5.60784787e-02, -5.41997552e-02, 4.78562862e-02};

    std::vector<float> s = {0.27039781,
                            0.19105849,
                            -0.06339942,
                            -0.65087199,
                            0.40867025,
                            0.05063812,
                            -0.14907975,
                            0.49018705,
                            -0.49197209,
                            0.33236548,
                            -0.39374301,
                            0.16012701,
                            0.06574871,
                            0.71606487,
                            -0.55201721,
                            -0.46427044};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4, 4, 1}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::float_type, {2, 3, 3, 3, 1}};
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        al,
        cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cos"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(cos_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("cos"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 0, 1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(cosh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data = {-1.0, 2.0, -3.0, 4.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cosh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return coshf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(cosh_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("cosh"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {-1.0, 2.0, -3.0, 4.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return coshf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convert_downcast_overflow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, 2 * std::numeric_limits<migraphx::half>::max());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(results_vector.begin(), results_vector.end(), [](const auto& x) {
        return x == std::numeric_limits<migraphx::half>::max();
    }));
}

TEST_CASE(convert_downcast_underflow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, 2 * std::numeric_limits<migraphx::half>::lowest());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(results_vector.begin(), results_vector.end(), [](const auto& x) {
        return x == std::numeric_limits<migraphx::half>::lowest();
    }));
}

TEST_CASE(convert_nan_upcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    std::vector<migraphx::half> data(4, std::numeric_limits<migraphx::half>::quiet_NaN());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4, -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_downcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {2, 2}};
    std::vector<double> data(4, std::numeric_limits<double>::quiet_NaN());
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4, -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_double_convert_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {2, 2}};
    std::vector<double> data(4, std::numeric_limits<double>::quiet_NaN());
    auto l   = mm->add_literal(migraphx::literal{s, data});
    auto f_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        f_l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convert_nan_convert_updown_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data(4, std::numeric_limits<float>::quiet_NaN());
    auto l   = mm->add_literal(migraphx::literal{s, data});
    auto f_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), l);
    auto h_l = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), f_l);
    mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), h_l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::all_of(
        results_vector.begin(), results_vector.end(), [](const auto& x) { return std::isnan(x); }));
}

TEST_CASE(convolution_backwards_1d)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3}};
    std::vector<float> x_data{0, 0.5, 1};
    std::vector<float> w_data{0.5, 0.5, 0.5};

    std::vector<float> gold{0, 0.25, 0.75, 0.75, 0.5};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(migraphx::make_op("convolution_backwards",
                                          {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                        x,
                        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2d)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{0,  1,  3, 3,  2,  3,  8,  15, 12, 7,  9,  21, 36,
                            27, 15, 9, 20, 33, 24, 13, 6,  13, 21, 15, 8};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(migraphx::make_op("convolution_backwards"), x, w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_3d)
{
    migraphx::shape s_1{migraphx::shape::float_type, {1, 1, 1, 2, 3}};
    migraphx::shape s_2{migraphx::shape::float_type, {1, 1, 3, 2, 3}};

    // clang-format off
    std::vector<float> x_data{0.8471, -0.4195, -2.2749, 1.2491, 0.1722, 0.3246};
    std::vector<float> w_data{
        0.6478, -0.1985, 0.0633, -0.3479, 2.7056, -0.1440,
        -1.1229, -0.7507, -1.3151, 0.8884, -0.1859, -0.3407,
        -1.1544, -1.5893, 1.6265, -1.4624, 0.3812, -1.5378
    };
    std::vector<float> gold{0.5488,  -0.4399, -1.3369, 0.4251,  -0.1439, 0.5145,  2.3015,  -0.2104,
                            -6.1482, 0.3482,  -0.4346, 3.3197,  0.1731,  0.8533,  -0.0467, -0.9512,
                            -0.1649, 1.7553,  2.2594,  2.9917,  -0.6500, -1.6612, -4.3680, 0.0957,
                            0.3482,  1.1097,  -0.0792, -0.1692, -0.1190, -0.1106, -0.9779, -0.8621,
                            4.6707,  2.9332,  -3.7001, -2.6808, -1.2476, 3.2475,  -0.4578, 4.0263,
                            -1.8267, 0.2243,  -2.3299, -0.1411, -0.4991};
    // clang-format on

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s_1, x_data});
    auto w   = mm->add_literal(migraphx::literal{s_2, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_padding1)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{8, 15, 12, 21, 36, 27, 20, 33, 24};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_padding2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{3., 8., 15., 12., 7., 9., 21., 36., 27., 15., 9., 20., 33., 24., 13.};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2stride)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> gold{0.,  0., 1., 1., 3.,  2.,  2.,  0.,  0.,  1., 1., 3.,  2.,
                            2.,  3., 3., 8., 5.,  12., 7.,  7.,  3.,  3., 7., 4.,  9.,
                            5.,  5., 9., 9., 20., 11., 24., 13., 13., 6., 6., 13., 7.,
                            15., 8., 8., 6., 6.,  13., 7.,  15., 8.,  8.};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2dilation)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> gold{0., 1., 2., 1.,  2.,  1.,  2.,  3.,  4.,  8., 4., 8., 4.,
                            5., 6., 8., 16., 8.,  16., 8.,  10., 3.,  4., 8., 4., 8.,
                            4., 5., 6., 8.,  16., 8.,  16., 8.,  10., 3., 4., 8., 4.,
                            8., 4., 5., 6.,  7.,  14., 7.,  14., 7.,  8.};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {2, 2}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_dyn_batch1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // clang-format off
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {3, 3}}};
    // clang-format on
    auto x = mm->add_parameter("x", s);
    auto w = mm->add_parameter("w", s);

    mm->add_instruction(migraphx::make_op("convolution_backwards"), x, w);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    params["w"] = migraphx::argument(input_fixed_shape, w_data.data());
    auto result = p.eval(params).back();

    std::vector<float> gold{0,  1,  3, 3,  2,  3,  8,  15, 12, 7,  9,  21, 36,
                            27, 15, 9, 20, 33, 24, 13, 6,  13, 21, 15, 8};
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_dyn_batch2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // clang-format off
    migraphx::shape x_shape{migraphx::shape::float_type,
                      {{1, 4}, {1, 1}, {5, 5}, {5, 5}}};
    // clang-format on
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape w_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> w_data(9, 1.);
    auto w = mm->add_literal(migraphx::literal{w_shape, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {2, 2}}, {"stride", {2, 2}}, {"dilation", {2, 2}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(25);
    std::iota(x_data.begin(), x_data.end(), 0.);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 5, 5}};
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();

    //clang-format off
    std::vector<float> gold{12.,  0., 21.,  0., 27.,  0., 33.,  0., 24.,  0., 0.,  0., 0.,   0.,
                            0.,   0., 0.,   0., 33.,  0., 54.,  0., 63.,  0., 72., 0., 51.,  0.,
                            0.,   0., 0.,   0., 0.,   0., 0.,   0., 63.,  0., 99., 0., 108., 0.,
                            117., 0., 81.,  0., 0.,   0., 0.,   0., 0.,   0., 0.,  0., 93.,  0.,
                            144., 0., 153., 0., 162., 0., 111., 0., 0.,   0., 0.,  0., 0.,   0.,
                            0.,   0., 72.,  0., 111., 0., 117., 0., 123., 0., 84.};
    //clang-format on

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(dequantizelinear)
{
    { /*uint8*/
        migraphx::shape xs{migraphx::shape::uint8_type, {1, 3, 3}};
        std::vector<uint8_t> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
        migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        migraphx::shape zs{migraphx::shape::uint8_type, {1, 3, 3}};
        std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::make_target("ref"));
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(9);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 2, 4, 10, 20, 100, 200, 300, 500};
        EXPECT(results_vector == gold);
    }

    { /*int8*/
        migraphx::shape xs{migraphx::shape::int8_type, {1, 3, 3}};
        std::vector<int8_t> xv = {-128, -100, -50, -1, 0, 1, 50, 100, 127};
        migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
        auto create_program   = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::make_target("ref"));
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(9);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{-256, -200, -100, -2, 0, 2, 100, 200, 254};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(dimensions_of_test0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4, {2, 4}}, {3, 3}, {4, 4}}};
    auto p1 = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), p1);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(24, 1.0);
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {2, 3, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(dimensions_of_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4, {1, 4}}, {3, 3}, {3, 8}, {3, 8}}};
    auto p1 = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 4}}), p1);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(48, 1.0);
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4, 4}};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();
    std::vector<int64_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {4, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(div_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1 = {-1.0f, 0.5f, 1.0f};
    std::vector<float> data2 = {1.0f, 2.0f, 4.0f};
    auto l1                  = mm->add_literal(migraphx::literal{s, data1});
    auto l2                  = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("div"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(data1.size());
    std::transform(data1.begin(), data1.end(), data2.begin(), gold.begin(), std::divides<float>());
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(div_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6, {3}}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("div"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1.0f, 0.5f, 1.0f};
    std::vector<float> y_data{1.0f, 2.0f, 4.0f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(x_data.size());
    std::transform(
        x_data.begin(), x_data.end(), y_data.begin(), gold.begin(), std::divides<float>());
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(elu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l      = mm->add_literal(migraphx::literal{s, {-1.0, 2.0, -3.0, 4.0}});
    float alpha = 0.5;
    mm->add_instruction(migraphx::make_op("elu", {{"alpha", alpha}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{elu(alpha, -1), elu(alpha, 2), elu(alpha, -3), elu(alpha, 4)};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(elu_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input  = mm->add_parameter("X", s);
    float alpha = 0.5;
    mm->add_instruction(migraphx::make_op("elu", {{"alpha", alpha}}), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1.0, 2.0, -3.0, 4.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{elu(alpha, -1), elu(alpha, 2), elu(alpha, -3), elu(alpha, 4)};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(equal_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto eq  = mm->add_instruction(migraphx::make_op("equal"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {true, false, false, false, true, false, true, false, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(equal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l0 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    auto l1 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1}});
    auto eq = mm->add_instruction(migraphx::make_op("equal"), l0, l1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {true, false, false, false, true, false, true, false, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(equal_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{6, 12, {9}}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto p0 = mm->add_parameter("l", s);
    auto p1 = mm->add_parameter("r", s);
    auto eq = mm->add_instruction(migraphx::make_op("equal"), p0, p1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        eq);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> l_data{1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> r_data{1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["l"] = migraphx::argument(input_fixed_shape0, l_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, r_data.data());
    auto result  = p.eval(params0).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {true, false, false, false, true, false, true, false, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(erf_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    std::vector<float> data = {0.73785057, 1.58165966, -0.43597795, -0.01677432};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("erf"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return erff(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(erf_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("erf"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {0.73785057, 1.58165966, -0.43597795, -0.01677432};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return erff(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(exp_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data{-1, 0, 1};
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("exp"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return expf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(exp_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("exp"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 0, 1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return expf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(floor_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data = {1.1, 1.5, 0.6, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("floor"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return floor(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(floor_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{5, 12};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("floor"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1.1, 1.5, 0.6, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return floor(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(fp16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {1}};
    migraphx::half a{1.5};
    migraphx::half b{2.5};
    migraphx::half c{4.0};
    auto l0 = mm->add_literal(migraphx::literal{s, {a}});
    auto l1 = mm->add_literal(migraphx::literal{s, {b}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{c};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(fp32_fp16_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        mm->add_instruction(migraphx::make_op("add"), l1, l2);
        return p;
    };

    auto test_case = [&](std::vector<std::string>&& op_names) {
        std::vector<float> gold_res = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
        auto p                      = create_program();
        migraphx::quantize_fp16(p, op_names);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res;
        result.visit([&](auto output) { res.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res, gold_res));
    };

    test_case({"all"});
    test_case({"add"});
}

TEST_CASE(gather_non_std_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data = {0.5f, 3.5f, 6.5f, 1.5f, 4.5f, 7.5f, 2.5f, 2.5f, 8.5f};
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto d = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{-3, -3, -1, -1};
        auto ind = mm->add_literal(migraphx::literal{s_indices, indices});
        auto td = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d);
        auto tind =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), ind);

        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), td, tind);
        auto result               = p.eval({}).back();
        std::vector<float> golden = {
            0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f, 0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        std::vector<float> res_data;
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }
}

TEST_CASE(gather_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{-3, -1};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 0;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = 1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
        std::vector<int> indices{0, 2};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data(4 * 5);
        std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f, 3.5f, 6.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3 * 3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{-3};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f, 3.5f, 6.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        std::vector<float> data(3);
        std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {3}};
        auto a0 = mm->add_literal(migraphx::literal{s, data});
        // scalar index
        migraphx::shape s_indices{migraphx::shape::int32_type};
        std::vector<int> indices{0};
        auto a1  = mm->add_literal(migraphx::literal{s_indices, indices});
        int axis = -1;
        mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), a0, a1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> golden = {0.5f};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(res_data, golden));
    }
}

TEST_CASE(gather_dyn_test0)
{
    // Dynamic data, static indices
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 5}, {3, 3}}};

    auto x = mm->add_parameter("x", s);
    std::vector<int> indices{1, 2};

    migraphx::shape s_ind{migraphx::shape::int32_type, {1, 2}};
    auto ind = mm->add_parameter("indices", s_ind);
    mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), x, ind);

    migraphx::shape sresult{migraphx::shape::int32_type, {{2, 5}, {1, 1}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == sresult);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 3}};
    migraphx::shape input_indices{migraphx::shape::int32_type, {1, 2}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"]       = migraphx::argument(input_fixed_shape, data.data());
    params["indices"] = migraphx::argument(input_indices, indices.data());
    auto result       = p.eval(params).back();

    std::vector<int> gold = {1, 2, 4, 5};
    std::vector<int> results_vector(2 * 1 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
    migraphx::shape sfinal{migraphx::shape::int32_type, {2, 1, 2}};
    EXPECT(result.get_shape() == sfinal);
}

TEST_CASE(gather_dyn_test1)
{
    // Dynamic data, dynamic indices
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 5}, {4, 4}}};

    auto x = mm->add_parameter("x", s);

    migraphx::shape s_ind{migraphx::shape::int32_type, {{1, 8, {7}}, {2, 3, {3}}}};
    auto ind = mm->add_parameter("indices", s_ind);
    mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x, ind);

    migraphx::shape sresult{migraphx::shape::int32_type, {{1, 8, {7}}, {2, 3, {3}}, {4, 4}}};
    EXPECT(p.get_output_shapes().back() == sresult);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {3, 4}};
    migraphx::shape input_indices_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{2, 0};
    migraphx::parameter_map params;

    std::vector<int> data(3 * 4);
    std::iota(data.begin(), data.end(), 0);
    params["x"]       = migraphx::argument(input_fixed_shape, data.data());
    params["indices"] = migraphx::argument(input_indices_shape, indices.data());
    auto result       = p.eval(params).back();

    std::vector<int> gold = {8, 9, 10, 11, 0, 1, 2, 3};
    std::vector<int> results_vector(1 * 2 * 4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, gold));
    migraphx::shape sfinal{migraphx::shape::int32_type, {1, 2, 4}};
    EXPECT(result.get_shape() == sfinal);
}

TEST_CASE(gathernd_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 2}};

        std::vector<float> data_vec(2 * 2);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{0, 0, 1, 1};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{0, 3};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 1}};

        std::vector<float> data_vec(2 * 2);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{1, 0};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{2, 3, 0, 1};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

        std::vector<float> data_vec(2 * 3 * 1);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{1, 0, 0, 1};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 3, 2, 3}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 2, 2}};

        std::vector<float> data_vec(2 * 3 * 2 * 3);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{0, 0, 0, 1, 0, 0, 0, 1};
        const int batch_dims = 1;

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1, 3}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 3, 2}};

        std::vector<float> data_vec(2 * 3 * 1 * 3);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{0, 0, 0, 1, 0, 2, 0, 2, 0, 1, 0, 0};
        const int batch_dims = 2;

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);

        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{0, 4, 8, 11, 13, 15};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        // k > r - batch_dims
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1, 3}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 3, 3}};

        std::vector<float> data_vec(2 * 3 * 1 * 3);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec(2 * 3 * 3, 0);
        const int batch_dims = 2;

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        EXPECT(test::throws([&] {
            mm->add_instruction(
                migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), data, indices);
        }));
    }
}

TEST_CASE(gathernd_dynamic0)
{
    // dynamic data, all dimensions fixed
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 2, {2}}, {3, 3}, {1, 1}}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};

    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic1)
{
    // dynamic data, dims not fixed
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 5, {2}}, {1, 5}, {1, 5}}};
    migraphx::shape is{migraphx::shape::int64_type, {2, 2, 1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic2)
{
    // dynamic both index and data
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {{2, 5, {2}}, {1, 5}, {1, 5}}};
    migraphx::shape is{migraphx::shape::int64_type, {{2, 5, {3}}, {2, 3, {3}}, {1, 1}}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic3)
{
    // dynamic index, static data and a batch_dims input
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape is{migraphx::shape::int64_type, {{2, 5, {3}}, {2, 3, {3}}, {1, 1}}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    int batch_dims{1};
    auto gathernd_op = migraphx::make_op("gathernd", {{"batch_dims", batch_dims}});
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3, 1}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {2, 2, 1}}; // index

    std::vector<float> data_vec(2 * 3 * 1);
    std::iota(data_vec.begin(), data_vec.end(), 0);
    std::vector<int64_t> indices_vec{1, 0, 0, 1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{1, 0, 3, 4};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(res_data, gold));
}

TEST_CASE(gathernd_dynamic4)
{
    // int(q) + r - k - batch_dims - 1 = 0 => returns a scalar
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape ds{migraphx::shape::float_type, {migraphx::shape::dynamic_dimension({2, 2})}};
    migraphx::shape is{migraphx::shape::int64_type, {1}};

    auto xdata  = mm->add_parameter("X", ds);
    auto xindex = mm->add_parameter("I", is);

    auto gathernd_op = migraphx::make_op("gathernd");
    auto gathernd    = mm->add_instruction(gathernd_op, xdata, xindex);

    mm->add_return({gathernd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2}}; // data
    migraphx::shape input_fixed_shape1{migraphx::shape::int64_type, {1}}; // index

    std::vector<float> data_vec(2);
    std::iota(data_vec.begin(), data_vec.end(), 4);
    std::vector<int64_t> indices_vec{1};
    params["X"] = migraphx::argument(input_fixed_shape0, data_vec.data());
    params["I"] = migraphx::argument(input_fixed_shape1, indices_vec.data());

    auto result = p.eval(params).back();
    std::vector<float> res_data{};
    std::vector<float> gold{5};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(res_data, gold));
}

TEST_CASE(gathernd_negative_index_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 1, 1}};

        std::vector<float> data_vec(2 * 2);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{-1, 0};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res_data{};
        std::vector<float> gold{2, 3, 0, 1};
        result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_range(res_data, gold));
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {2, 2}};
        migraphx::shape is{migraphx::shape::int64_type, {2, 1, 1}};

        std::vector<float> data_vec(2 * 2);
        std::iota(data_vec.begin(), data_vec.end(), 0);
        std::vector<int64_t> indices_vec{-3, 0};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, indices_vec});

        mm->add_instruction(migraphx::make_op("gathernd"), data, indices);
        p.compile(migraphx::make_target("ref"));

        EXPECT(test::throws([&] { p.eval({}); }));
    }
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(globalavgpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6}, {2, 6, {2}}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::average}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(globallppool_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op     = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    auto lens   = s.lens();
    op.lengths  = {lens[2], lens[3]};
    op.lp_order = 2;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5477225575051662, 1.307669683062202, 0.9327379053088815};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(globallppool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s =
        migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6, {2}}, {2, 6, {2}}}};
    auto x = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::lpnorm}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5477225575051662, 1.307669683062202, 0.9327379053088815};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s =
        migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6, {2}}, {2, 6, {2}}}};
    auto x = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(greater_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto gr  = mm->add_instruction(migraphx::make_op("greater"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, false, true, false, true, false, true, false};
    EXPECT(results_vector == gold);
}

TEST_CASE(greater_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l0 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    auto l1 =
        mm->add_literal(migraphx::literal{s, {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1}});
    auto gr = mm->add_instruction(migraphx::make_op("greater"), l0, l1);
    auto r  = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, true, false, true, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(greater_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{8, 10, {9}}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    auto gr    = mm->add_instruction(migraphx::make_op("greater"), left, right);
    auto r     = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        gr);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> left_data{1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> right_data{1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, true, false, true, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(identity_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<int> data{1, 2, 3, 4};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("identity"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::equal(data.begin(), data.end(), results_vector.begin()));
}

TEST_CASE(identity_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {2, 4}}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("identity"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<int> input_data{1, 2, 3, 4};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::int32_type, {2, 2}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<int> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(std::equal(input_data.begin(), input_data.end(), results_vector.begin()));
}

TEST_CASE(if_literal_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);

        migraphx::shape s{migraphx::shape::float_type, {5}};

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        else_mod->add_return({l2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_param_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", cond_s);
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        auto x                   = mm->add_parameter("x", ds);
        auto y                   = mm->add_parameter("y", ds);
        std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
        auto l2                  = mm->add_literal(migraphx::literal(ds, data2));
        auto sum                 = mm->add_instruction(migraphx::make_op("add"), x, l2);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
        auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
        auto tx                  = then_mod->add_parameter("x", ds);
        auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), tx, l1);
        then_mod->add_return({a1});

        auto* else_mod = p.create_module("If_0_else");
        auto ey        = else_mod->add_parameter("y", ds);
        auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), ey, sum);
        else_mod->add_return({a2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond, x, y}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data_x(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data_x.data());
        std::vector<float> data_y(ds.elements(), 2);
        m["y"] = migraphx::argument(ds, data_y.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });
        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {
            1.384804, -0.77947998, 0.54622501, 1.477438, -0.063330054, -0.12892997};
        auto ret = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {
            1.483906, 2.720788, 3.0736079, 0.84447598, 4.0433998, 4.04884};
        auto ret = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(if_pl_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape cond_s{migraphx::shape::bool_type};
        migraphx::shape s{migraphx::shape::float_type, {5}};
        auto cond = mm->add_parameter("cond", cond_s);
        auto x    = mm->add_parameter("x", s);

        auto* then_mod           = p.create_module("If_0_if");
        std::vector<float> data1 = {1, 2, 3, 4, 5};
        auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
        then_mod->add_return({l1, x});

        auto* else_mod           = p.create_module("If_0_else");
        std::vector<float> data2 = {5, 4, 3, 2, 1};
        auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
        auto s2                  = else_mod->add_instruction(migraphx::make_op("add"), x, l2);
        else_mod->add_return({s2, l2});

        auto ret     = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto outline = mm->add_outline(s);
        auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({outline, r});

        return p;
    };

    auto run_prog = [&](bool cond) {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        std::vector<char> c_data = {static_cast<char>(cond)};
        migraphx::shape cs{migraphx::shape::bool_type};
        migraphx::parameter_map m;
        m["cond"] = migraphx::argument(cs, c_data.data());
        migraphx::shape ds{migraphx::shape::float_type, {5}};
        std::vector<float> data(ds.elements(), 1);
        m["x"] = migraphx::argument(ds, data.data());

        auto res = p.eval(m).back();
        std::vector<float> ret;
        res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });

        return ret;
    };

    // then branch
    {
        std::vector<float> gold_ret = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto ret                    = run_prog(true);
        EXPECT(gold_ret == ret);
    }

    // else branch
    {
        std::vector<float> gold_ret = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f};
        auto ret                    = run_prog(false);
        EXPECT(gold_ret == ret);
    }
}

TEST_CASE(isnan_test)
{
    // float test
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto nan_val             = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> data0 = {1.2, 5.2, nan_val, nan_val, 0., 100.};
        auto l1                  = mm->add_literal(migraphx::literal{s, data0});
        mm->add_instruction(migraphx::make_op("isnan"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> correct = {0, 0, 1, 1, 0, 0};
        EXPECT(migraphx::verify::verify_range(results_vector, correct));
    }

    // half test
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        auto nan_val = std::numeric_limits<migraphx::half>::quiet_NaN();
        migraphx::half a{1.2};
        migraphx::half b{5.2};
        std::vector<migraphx::half> data0 = {a, b, nan_val, nan_val, b, a};
        auto l1                           = mm->add_literal(migraphx::literal{s, data0});
        mm->add_instruction(migraphx::make_op("isnan"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> correct = {0, 0, 1, 1, 0, 0};
        EXPECT(migraphx::verify::verify_range(results_vector, correct));
    }
}

TEST_CASE(isnan_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {3, 8}}};
    auto input   = mm->add_parameter("X", s);
    auto nan_val = std::numeric_limits<float>::quiet_NaN();
    mm->add_instruction(migraphx::make_op("isnan"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1.2, 5.2, nan_val, nan_val, 0., 100.};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> correct = {0, 0, 1, 1, 0, 0};
    EXPECT(migraphx::verify::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_no_pad_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, input));
}

TEST_CASE(im2col_3x3_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {4, 4};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> correct = {0, 1, 2, 4, 5, 6,  8,  9,  10, 1, 2, 3, 5, 6,  7,  9,  10, 11,
                                4, 5, 6, 8, 9, 10, 12, 13, 14, 5, 6, 7, 9, 10, 11, 13, 14, 15};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_stride_2_no_pad_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {6, 6};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{2, 2};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> correct = {0,  1,  2,  6,  7,  8,  12, 13, 14, 2,  3,  4,
                                8,  9,  10, 14, 15, 16, 12, 13, 14, 18, 19, 20,
                                24, 25, 26, 14, 15, 16, 20, 21, 22, 26, 27, 28};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, correct));
}

TEST_CASE(im2col_3x3_with_channels_identity_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {3, 3};
    std::vector<std::size_t> padding{0, 0};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 2;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, input));
}

TEST_CASE(im2col_3x3_with_padding_test)
{
    std::size_t f[2]    = {3, 3};
    std::size_t size[2] = {2, 2};
    std::vector<std::size_t> padding{1, 1};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t channels = 1;

    std::vector<int32_t> weights(channels * f[0] * f[1]);
    std::vector<int32_t> input(channels * size[0] * size[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_image{migraphx::shape::int32_type, {1, channels, size[0], size[1]}};
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_image   = mm->add_literal(migraphx::literal{s_image, input});
    auto l_weights = mm->add_literal(migraphx::literal{s_weights, weights});
    mm->add_instruction(
        migraphx::make_op("im2col",
                          {{"padding", padding}, {"stride", stride}, {"dilation", dilation}}),
        l_image,
        l_weights);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int> correct = {0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0,
                                0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0};

    std::size_t col_height = (size[0] - f[0] + 2 * padding[0]) / stride[0] + 1;
    std::size_t col_width  = (size[1] - f[1] + 2 * padding[1]) / stride[1] + 1;
    std::vector<float> results_vector(channels * f[0] * f[1] * col_height * col_width);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, correct));
}

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto img           = mm->add_literal(migraphx::literal{s,
                                                 {0.2,
                                                  0.3,
                                                  0.5,
                                                  0.4,

                                                  0.7,
                                                  0.8,
                                                  0.1,
                                                  0.9,

                                                  0.15,
                                                  0.25,
                                                  0.35,
                                                  0.45}});
    auto scale_val     = mm->add_literal(2.f);
    auto scaled_tensor = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), scale_val);
    auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), img, scaled_tensor);
    auto bias_vals  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), bias_vals);
    mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.41,
                               0.61,
                               1.01,
                               0.81,

                               1.42,
                               1.62,
                               0.22,
                               1.82,

                               0.33,
                               0.53,
                               0.73,
                               0.93};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(leaky_relu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 0.01}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.01f, 0.f, 1.f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(less_brcst_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {3, 3}};
    auto l0 =
        mm->add_literal(migraphx::literal{s0, {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0}});
    migraphx::shape s1{migraphx::shape::float_type, {3, 1}};
    auto l1  = mm->add_literal(migraphx::literal{s1, {1.1, -1.5, 0.0}});
    auto bl1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), l1);
    auto le  = mm->add_instruction(migraphx::make_op("less"), l0, bl1);
    auto r   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, false, true, false, false, false, false, false, true};
    EXPECT(results_vector == gold);
}

TEST_CASE(less_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    std::vector<float> data1 = {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> data2 = {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    auto l0                  = mm->add_literal(migraphx::literal{s, data1});
    auto l1                  = mm->add_literal(migraphx::literal{s, data2});
    auto le                  = mm->add_instruction(migraphx::make_op("less"), l0, l1);
    auto r                   = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](float n1, float n2) -> bool {
            return n1 < n2;
        });
    EXPECT(results_vector == gold);
}

TEST_CASE(less_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{8, 10, {9}}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    auto le    = mm->add_instruction(migraphx::make_op("less"), left, right);
    auto r     = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> left_data  = {1.1, 1.5, 0.1, -1.1, -1.5, -0.6, 0.0, 2.0, -2.0};
    std::vector<float> right_data = {1.1, 1.6, -0.1, -1.2, -1.5, -0.7, 0.0, 2.3, -2.1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<bool> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(left_data.size());
    std::transform(left_data.begin(),
                   left_data.end(),
                   right_data.begin(),
                   gold.begin(),
                   [](float n1, float n2) -> bool { return n1 < n2; });
    EXPECT(results_vector == gold);
}

TEST_CASE(log_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {1, 2, 3};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("log"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return logf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(log_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("log"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return logf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_and_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_and"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data2.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 and n2;
        });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_and_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6, {4}}};
    migraphx::shape s{migraphx::shape::bool_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    mm->add_instruction(migraphx::make_op("logical_and"), left, right);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> left_data{1, 0, 1, 0};
    std::vector<char> right_data{1, 1, 0, 0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::bool_type, {4}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(left_data.size());
    std::transform(left_data.begin(),
                   left_data.end(),
                   right_data.begin(),
                   gold.begin(),
                   [](bool n1, bool n2) -> bool { return n1 and n2; });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_or_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_or"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 or n2;
        });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_or_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6, {4}}};
    migraphx::shape s{migraphx::shape::bool_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    mm->add_instruction(migraphx::make_op("logical_or"), left, right);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> left_data{1, 0, 1, 0};
    std::vector<char> right_data{1, 1, 0, 0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::bool_type, {4}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold(left_data.size());
    std::transform(left_data.begin(),
                   left_data.end(),
                   right_data.begin(),
                   gold.begin(),
                   [](bool n1, bool n2) -> bool { return n1 or n2; });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_xor_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_xor"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, true, false};
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 ^ n2;
        });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logical_xor_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6, {4}}};
    migraphx::shape s{migraphx::shape::bool_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    mm->add_instruction(migraphx::make_op("logical_xor"), left, right);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> left_data{1, 0, 1, 0};
    std::vector<char> right_data{1, 1, 0, 0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::bool_type, {4}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, true, false};
    std::transform(left_data.begin(),
                   left_data.end(),
                   right_data.begin(),
                   gold.begin(),
                   [](bool n1, bool n2) -> bool { return n1 ^ n2; });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(logsoftmax_test_axis_0)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.135261, -2.843968, -0.659995, -0.488413, -1.051857, -2.812936, -0.250956, -0.353985,
        -1.155980, -0.603651, -0.211969, -0.175371, -1.336552, -3.885010, -1.871544, -0.837083,
        -0.887745, -0.433338, -1.158864, -4.911197, -1.147972, -0.666711, -0.996874, -0.981418,
        -0.851145, -0.853988, -0.858112, -2.067420, -0.059956, -0.727436, -0.950881, -0.429689,
        -0.061906, -1.505332, -1.210277, -0.377970, -0.791448, -1.655428, -1.827253, -0.304828,
        -0.020762, -0.167101, -0.567346, -0.530319, -1.045094, -0.376648, -0.007391, -0.381670,
        -0.720302, -0.460499, -0.469651, -0.556740, -0.554628, -0.551582};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_1)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.550468, -2.132973, -1.549746, -0.650533, -1.051529, -2.248570, -0.141017, -2.028357,
        -1.947730, -1.511324, -0.166597, -0.379726, -1.965689, -1.172109, -1.475721, -2.700831,
        -1.537011, -0.658754, -1.596017, -3.353137, -2.266743, -1.084197, -1.076214, -0.406712,
        -2.743019, -0.425526, -1.079083, -2.139486, -1.270584, -1.024088, -1.154231, -3.201762,
        -0.888957, -0.532855, -3.103583, -1.221339, -1.355980, -3.531678, -1.438510, -0.975194,
        -0.080261, -1.162697, -1.568557, -1.398519, -1.322129, -0.470660, -0.370953, -0.907343,
        -1.179017, -3.312239, -1.286363, -1.586076, -0.345100, -0.824173};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_2)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.495957, -1.031212, -0.245531, -2.013726, -1.339125, -2.465619, -1.356652, -0.964037,
        -2.019250, -0.214522, -0.289569, -0.234392, -2.086591, -2.684439, -2.851651, -2.674176,
        -1.697424, -1.889155, -0.401029, -3.064586, -1.173030, -1.306912, -2.177020, -0.834262,
        -2.818177, -0.174415, -1.361105, -1.024571, -0.106766, -1.167645, -1.072650, -2.576522,
        -0.569261, -1.207483, -3.679894, -2.095913, -0.504264, -3.039291, -1.290559, -1.156812,
        -0.126453, -0.551493, -2.506384, -2.646261, -1.905195, -0.206994, -0.191369, -0.959754,
        -1.948685, -3.671233, -0.875521, -3.111952, -1.905644, -1.6076011};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 2;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_3)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.336904, -3.475825, -1.366154, -0.279366, -2.208430, -2.010934, -0.225511, -2.436562,
        -2.167785, -1.572415, -1.784104, -0.470789, -1.067459, -1.801948, -0.711023, -2.307197,
        -1.467087, -0.400681, -0.426983, -3.740518, -1.127681, -1.078919, -2.599005, -0.534965,
        -2.561400, -0.567617, -1.033025, -2.097713, -0.520463, -1.262245, -1.763230, -2.607658,
        -0.281299, -0.814243, -2.627210, -0.724131, -0.655704, -2.123055, -1.018163, -2.480634,
        -0.382599, -1.451479, -1.843102, -0.915303, -0.818078, -1.316929, -0.508875, -2.033541,
        -1.487672, -2.417791, -0.378360, -2.568531, -0.569794, -1.028032};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 3;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(lppool_l1_norm_test)
{
    // L1 norm test
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op     = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    op.lengths  = {2};
    op.padding  = {0};
    op.stride   = {1};
    op.lp_order = 1;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5, 0.6, 0.5, 1.3, 1.4, 1.0, 0.8, 0.8, 0.7};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

// TODO: this tests compliance with a oneDNN rule and a feature that's commented out
// in pooling.hpp
// TEST_CASE(lppool_l1_norm_err_test)
// {
//     // padding too large for kernel size
//     migraphx::program p;
//     auto* mm    = p.get_main_module();
//     auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 5}};
//     auto op     = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
//     op.lengths  = {3};
//     op.padding  = {2};
//     op.stride   = {1};
//     op.lp_order = 1;

//     std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7};
//     auto l0 = mm->add_literal(migraphx::literal{s, data});
//     EXPECT(test::throws([&] {
//             mm->add_instruction(op, l0);
//         }));
// }

TEST_CASE(lppool_l2_norm_test)
{
    // L2 norm test
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op     = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    op.lengths  = {2};
    op.padding  = {0};
    op.stride   = {1};
    op.lp_order = 2;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.36055512754639896,
                            0.447213595499958,
                            0.4123105625617661,
                            0.9433981132056605,
                            1.0295630140987,
                            0.9055385138137417,
                            0.7071067811865475,
                            0.7071067811865475,
                            0.6082762530298219};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(lppool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::lpnorm},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.36055512754639896,
                            0.447213595499958,
                            0.4123105625617661,
                            0.9433981132056605,
                            1.0295630140987,
                            0.9055385138137417,
                            0.7071067811865475,
                            0.7071067811865475,
                            0.6082762530298219};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(lrn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 5, 1, 1}};
    auto l = mm->add_literal(migraphx::literal{s, {-2.0f, 1.0f, 0.f, 1.0f, 2.0f}});
    mm->add_instruction(
        migraphx::make_op("lrn", {{"alpha", 0.0001}, {"beta", 0.75}, {"bias", 1}, {"size", 5}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(5);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2 / 1.000075, 1 / 1.00009, 0 / 1.000145, 1 / 1.00009, 2 / 1.000075};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(max_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_max = mm->add_instruction(migraphx::make_op("max"), l0, l1);
    mm->add_instruction(migraphx::make_op("max"), curr_max, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{7, 8, 9};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(max_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x        = mm->add_parameter("x", s);
    auto y        = mm->add_parameter("y", s);
    auto z        = mm->add_parameter("z", s);
    auto curr_max = mm->add_instruction(migraphx::make_op("max"), x, y);
    mm->add_instruction(migraphx::make_op("max"), curr_max, z);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{1, 4, 3};
    std::vector<float> y_data{2, 8, 6};
    std::vector<float> z_data{7, 5, 9};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    params0["z"] = migraphx::argument(input_fixed_shape0, z_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{7, 8, 9};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -2.1314404,  -1.63041711, 1.54562736,  1.04625261,  -1.42931843, -0.48703974, 0.4065806,
        -0.1524526,  1.30775225,  0.45538983,  -0.06631992, -1.75332725, 1.33493888,  0.47327688,
        0.36873096,  1.18358743,  -0.34640595, 1.22098756,  0.01946825,  -0.20238149, 0.43348005,
        -0.67991608, -0.83041084, 0.93537551,  0.70241445,  -0.5654031,  -1.30899191, -0.26735824,
        -0.52444768, 1.99097753,  1.86504853,  -0.26506025, 0.26236168,  0.43763575,  0.95300823,
        -1.02733946, -0.74655169, -0.5374338,  -0.28901565, -0.59789604, 0.5310151,   0.99125904,
        0.40609556,  -1.57175648, 0.22031412,  1.45862222,  0.53217483,  1.39087725,  1.00170159,
        -0.87175864, -1.7204628,  -1.72008383, -0.38656762, -0.01443311, 1.46645272,  -1.39995027,
        0.22505587,  -0.43461126, -0.05511411, -0.79950953, -0.01439556, 0.08795211,  1.18943918,
        -0.84079367, -1.73383629, -0.55662078, -0.30626822, -0.67339015, 0.44179603,  0.54316711,
        0.40899998,  -0.27831686, -1.11900508, -0.0881724,  0.35483059,  2.36277103,  -0.04765317,
        -0.36865309, 0.73814237,  1.47151589,  1.36546791,  -0.32649881, -1.0517807,  2.24768877,
        0.68883753,  0.58646208,  -0.91017133, -0.50462508, -0.4013325,  -0.72348958, -0.47368807,
        0.35285577,  -1.01817429, -0.5152272,  0.60321307,  0.43521205,  -0.23733577, 0.66427642,
        0.82949388,  0.82443929,  0.71550399,  0.34561086,  0.68570769,  -0.40718508, -1.20350206,
        0.15793853,  -2.31013632, -0.07934658, -0.09348056, 0.36576006,  2.46601582,  0.11090943,
        0.9144392,   0.56759721,  -0.22112127, -0.21955389, 0.72474903,  -1.28448462, 1.53285873,
        0.37437943,  0.31409341,  1.95433736,  0.91620457,  0.86205518,  1.24365854,  0.19248386,
        0.22526583,  0.13462132,  -0.27561715, -2.06446075, -0.02306402, -1.38278747, 1.1411345,
        1.31293464,  -1.86041689, 1.06763375,  -0.26541466, 1.4545635,   1.11430049,  -0.66491818,
        0.87101674,  0.67768967,  -1.02062869, -1.05031872, -2.2764678,  -2.0200038,  0.37592548,
        -0.26701379, -0.83388507, 0.19403623,  1.00968623,  0.11020003,  1.16736257,  -1.1160326,
        0.47346735,  0.6126079,   -0.19135755, 1.33624589,  -0.29802522, -0.57873946, -1.06555879,
        -0.20686582, 1.36892557,  -0.19937795, 0.8649236,   -1.40126073, 1.53441942,  0.34682792,
        -1.31724346, -1.32898355, 2.40126371,  0.07845283,  1.35732043,  -0.63678312, 0.39429256,
        -1.36487007, -0.31026676, -0.44981545, -0.28994772, -0.14657612, -1.75206447, -0.70612341,
        1.20071781,  -1.64647579, -0.7133292,  0.88494766,  0.52119428,  -2.77387547, 2.07681108,
        -0.90133125, 0.2847338,   0.6174528,   -0.20616426, -0.64263535, -1.08496261, 0.54275119,
        -0.88503587, 0.6629802,   1.47319221,  -1.05829155, -0.97027361, -0.93187737, -1.39954746,
        -0.52359426, -0.14743951, 1.51522756,  0.2078452,   -1.28156149, -1.19363916, -0.78680223,
        -0.89094824, 1.30212069,  -0.77974445, -0.58411664, 0.48764706,  -0.67132682};
    std::vector<float> c = {1.33493888, 1.54562736, 1.22098756, 1.33493888, 1.18358743, 1.99097753,
                            1.00170159, 1.45862222, 1.39087725, 1.46645272, 1.18943918, -0.01443311,
                            1.47151589, 2.36277103, 2.24768877, 0.68883753, 0.82949388, 0.71550399,
                            1.95433736, 2.46601582, 1.53285873, 1.95433736, 1.06763375, 1.4545635,
                            1.33624589, 1.16736257, 0.6126079,  1.36892557, 2.40126371, 1.53441942,
                            0.52119428, 2.07681108, 0.88494766, 1.51522756, 0.54275119, 0.6629802};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 6, 6}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {0, 0}},
                                           {"stride", {2, 2}},
                                           {"lengths", {3, 2}}}),
                        al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(36);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, c));
}

TEST_CASE(maxpool_pad_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {-6, -5, -4, -3, -5, -1, 0, 1, 2, 3, 4, 5};
    std::vector<float> c = {-4, -3, -4, -1, 2, 3, 4, 5};
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 3, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {1, 1}},
                                           {"stride", {2, 2}},
                                           {"lengths", {3, 2}}}),
                        al);

    //   * *  *  *                                           * *  *  *
    //   * -6 -5 *                                           * 0  1  *
    //   * -4 -3 *      padding will look like this          * 2  3  *
    //   * -5 -1 *                  and this                 * 4  5  *
    //   * *  *  *      The * values are actually -INF       * *  *  *

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, c));
}

TEST_CASE(maxpool_rank3_test0)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths = {2};
    op.padding = {0};
    op.stride  = {1};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.3, 0.4, 0.4, 0.8, 0.9, 0.9, 0.7, 0.7, 0.6};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_test1)
{
    // 1D case 2, input is 3D
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths = {2};
    op.padding = {0};
    op.stride  = {2};

    std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, -0.6186, -0.9618,
                            0.6022, -0.1912, 1.1925,  0.5493,  0.1692,  -0.8039, -1.0281,
                            0.9907, 0.477,   1.5001,  -1.1603, -1.361,  1.2556};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4975, -0.0405, -0.6186, 0.6022, 0.5493, -0.8039, 1.5001, -1.1603};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_ceil_test)
{
    // 1D case 2, input is 3D, ceil mode
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {2};
    op.ceil_mode = true;

    // clang-format off
    std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, 
                        -0.6186, -0.9618, 0.6022, -0.1912, 1.1925,
                        0.5493,  0.1692,  -0.8039, -1.0281, 0.9907, 
                        0.477,   1.5001,  -1.1603, -1.361,  1.2556};
    // clang-format on
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold{0.4975, -0.0405, -0.1227, -0.6186,
                            0.6022, 1.1925, 0.5493, -0.8039,
                            0.9907, 1.5001, -1.1603, 1.2556};
    // clang-format on
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_rank5_test)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths = {2, 2, 2};
    op.padding = {0, 0, 0};
    op.stride  = {2, 2, 2};

    std::vector<float> data{
        -2.8029, 0.5861,  0.7015,  0.1297,  -1.44,   -1.9472, 0.7812,  2.408,   -0.3145, 0.3405,
        -0.9146, 0.0624,  1.5064,  -0.8345, 1.7977,  1.8949,  1.0073,  -0.2102, -0.042,  -0.7146,
        0.6227,  -0.5263, -2.2598, 0.1713,  0.449,   0.5303,  -0.8622, -0.5691, 0.907,   -0.0569,
        -1.5348, -0.4109, -0.1461, -0.5445, 0.4266,  0.2282,  1.3655,  -2.1519, 0.6068,  -0.2001,
        -0.4702, 0.3864,  1.7083,  0.9096,  0.4286,  -1.8866, 0.7034,  0.0293,  1.4587,  0.7672,
        -2.8614, 0.8124,  -0.053,  1.0449,  0.845,   -0.0131, 0.1139,  -0.859,  -1.2681, -0.6337,
        -0.4644, 0.1938,  0.2889,  0.9035,  0.7118,  -0.5767, 0.4577,  -0.0549, 0.2237,  0.5756,
        0.0677,  -0.0223, -0.329,  0.2364,  2.7666,  -0.7417, -1.3196, -0.2655, 0.1698,  -0.1777,
        -0.9427, 2.6859,  -0.7501, 0.5175,  1.0029,  -2.6436, -0.4388, -1.2348, -0.1539, -0.6229,
        -0.4136, 0.5085,  0.4136,  -0.6439, -1.1953, -0.406,  -0.0195, 0.1869,  -0.8664, 1.1364,
        0.5041,  0.0647,  0.1941,  -1.0819, -0.4629, -0.5107, 0.3612,  -0.3583};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5064, 1.3655, 0.9035, 2.6859};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(maxpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.3, 0.4, 0.4, 0.8, 0.9, 0.9, 0.7, 0.7, 0.6};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(min_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {1, 4, 3}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2, 8, 6}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_min = mm->add_instruction(migraphx::make_op("min"), l0, l1);
    mm->add_instruction(migraphx::make_op("min"), curr_min, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 4, 3};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(min_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x        = mm->add_parameter("x", s);
    auto y        = mm->add_parameter("y", s);
    auto z        = mm->add_parameter("z", s);
    auto curr_min = mm->add_instruction(migraphx::make_op("min"), x, y);
    mm->add_instruction(migraphx::make_op("min"), curr_min, z);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{1, 4, 3};
    std::vector<float> y_data{2, 8, 6};
    std::vector<float> z_data{7, 5, 9};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    params0["z"] = migraphx::argument(input_fixed_shape0, z_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 4, 3};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(fmod_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {-7, 8, -3}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2, 4, 6}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7, 5, 9}});
    auto curr_mod = mm->add_instruction(migraphx::make_op("fmod"), l0, l1);
    mm->add_instruction(migraphx::make_op("fmod"), curr_mod, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-1, 0, -3};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(fmod_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x        = mm->add_parameter("x", s);
    auto y        = mm->add_parameter("y", s);
    auto z        = mm->add_parameter("z", s);
    auto curr_mod = mm->add_instruction(migraphx::make_op("fmod"), x, y);
    mm->add_instruction(migraphx::make_op("fmod"), curr_mod, z);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-7, 8, -3};
    std::vector<float> y_data{2, 4, 6};
    std::vector<float> z_data{7, 5, 9};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    params0["z"] = migraphx::argument(input_fixed_shape0, z_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-1, 0, -3};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(fmod_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {-7.2f, 8.5f, -3.3f}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2.0f, 4.0f, 6.0f}});
    auto l2       = mm->add_literal(migraphx::literal{s, {7.0f, 5.0f, 9.0f}});
    auto curr_mod = mm->add_instruction(migraphx::make_op("fmod"), l0, l1);
    mm->add_instruction(migraphx::make_op("fmod"), curr_mod, l2);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-1.2f, 0.5f, -3.3f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(mod_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {-3, 8, -7}});
    auto l1       = mm->add_literal(migraphx::literal{s, {3, 3, 3}});
    auto l2       = mm->add_literal(migraphx::literal{s, {10, 2, 9}});
    auto curr_mod = mm->add_instruction(migraphx::make_op("mod"), l0, l1);
    mm->add_instruction(migraphx::make_op("mod"), curr_mod, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 2};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(mod_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x        = mm->add_parameter("x", s);
    auto y        = mm->add_parameter("y", s);
    auto z        = mm->add_parameter("z", s);
    auto curr_mod = mm->add_instruction(migraphx::make_op("mod"), x, y);
    mm->add_instruction(migraphx::make_op("mod"), curr_mod, z);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-3, 8, -7};
    std::vector<float> y_data{3, 3, 3};
    std::vector<float> z_data{10, 2, 9};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    params0["z"] = migraphx::argument(input_fixed_shape0, z_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 2};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(mod_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l0       = mm->add_literal(migraphx::literal{s, {-3.0f, 8.5f, -7.0f}});
    auto l1       = mm->add_literal(migraphx::literal{s, {2.0f, 3.0f, 3.0f}});
    auto l2       = mm->add_literal(migraphx::literal{s, {3.0f, 3.0f, 4.0f}});
    auto curr_mod = mm->add_instruction(migraphx::make_op("mod"), l0, l1);
    mm->add_instruction(migraphx::make_op("mod"), curr_mod, l2);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0f, 2.5f, 2.0f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(mul_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1{-1, 0, 1};
    std::vector<float> data2{1, 2, 3};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("mul"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(data1.size());
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](float n1, float n2) -> float {
            return n1 * n2;
        });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(mul_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("mul"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 1};
    std::vector<float> y_data{1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(x_data.size());
    std::transform(x_data.begin(),
                   x_data.end(),
                   y_data.begin(),
                   gold.begin(),
                   [](float n1, float n2) -> float { return n1 * n2; });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    auto l1 = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", l1->get_shape().lens()}}),
                        l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -3);
    EXPECT(output(1, 0) == -2);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(multibroadcast_2in_static_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    auto l1 = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("multibroadcast"), l2, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -3);
    EXPECT(output(1, 0) == -2);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(multibroadcast_2in_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {{2, 4}, {2, 2}}};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    std::vector<int32_t> b_data{-2, -3};
    auto l1 = mm->add_parameter("a", a_shape);
    auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
    mm->add_instruction(migraphx::make_op("multibroadcast"), l2, l1);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> a_data{0, 0, 0, 0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 2}};
    params0["a"] = migraphx::argument(input_fixed_shape0, a_data.data());
    auto result  = p.eval(params0).back();
    auto output  = result.get<int32_t>();
    EXPECT(output(0, 0) == -2);
    EXPECT(output(0, 1) == -3);
    EXPECT(output(1, 0) == -2);
    EXPECT(output(1, 1) == -3);
}

TEST_CASE(multibroadcast_3in_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int32_type, {{2, 4}, {2, 2}}};
    migraphx::shape b_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape c_shape{migraphx::shape::int32_type, {{1, 4, {2, 4}}, {2, 4}, {2, 2}}};
    auto l1 = mm->add_parameter("a", a_shape);
    std::vector<int32_t> b_data{-2, -3};
    auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
    auto l3 = mm->add_parameter("c", c_shape);
    mm->add_instruction(migraphx::make_op("multibroadcast"), l2, l1, l3);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> a_data(4, 0);
    std::vector<int32_t> c_data(8, 0);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape_a{migraphx::shape::float_type, {2, 2}};
    migraphx::shape input_fixed_shape_c{migraphx::shape::float_type, {2, 2, 2}};
    params["a"] = migraphx::argument(input_fixed_shape_a, a_data.data());
    params["c"] = migraphx::argument(input_fixed_shape_c, c_data.data());
    auto result = p.eval(params).back();
    auto output = result.get<int32_t>();
    EXPECT(output(0, 0, 0) == -2);
    EXPECT(output(0, 0, 1) == -3);
    EXPECT(output(0, 1, 0) == -2);
    EXPECT(output(0, 1, 1) == -3);
    EXPECT(output(1, 0, 0) == -2);
    EXPECT(output(1, 0, 1) == -3);
    EXPECT(output(1, 1, 0) == -2);
    EXPECT(output(1, 1, 1) == -3);
}

TEST_CASE(multinomial_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size = 100000;
    float seed         = 0.0f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> rand_samples(sample_size);
    std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
    migraphx::shape rs{migraphx::shape::float_type, {1, sample_size}};
    auto rs_lit = mm->add_literal(migraphx::literal{rs, rand_samples});

    migraphx::shape s{migraphx::shape::float_type, {1, 5}};
    std::vector<int> dist{15, 25, 15, 25, 20};
    std::vector<float> data(5);
    std::transform(dist.begin(), dist.end(), data.begin(), [&](auto d) { return std::log(d); });
    auto input = mm->add_literal(migraphx::literal(s, data));

    auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);
    auto mb_maxes =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 5}}}), maxes);
    auto cdf = mm->add_instruction(migraphx::make_op("sub"), input, mb_maxes);
    cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
    cdf      = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

    mm->add_instruction(migraphx::make_op("multinomial"), cdf, rs_lit);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int32_t> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    std::vector<int> res_dist(5, 0);
    for(const auto& r : result_vec)
        res_dist[r]++;
    auto dist_sum     = std::accumulate(dist.begin(), dist.end(), 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);
    std::transform(dist.begin(), dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });
    EXPECT(migraphx::verify::verify_range(norm, res_norm, 100000));
}

TEST_CASE(neg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 1.3f, -1.2f, 0.0f, -100.f, 200.f};
    auto input              = mm->add_literal(migraphx::literal(s, data));
    auto ret                = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(gold.begin(), gold.end(), gold.begin(), std::negate<float>());
    EXPECT(migraphx::verify::verify_range(result_vector, gold));
}

TEST_CASE(neg_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {3, 3}}};
    auto input = mm->add_parameter("X", s);
    auto ret   = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {1.0f, 1.3f, -1.2f, 0.0f, -100.f, 200.f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());
    auto result  = p.eval(params0).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = a;
    std::transform(gold.begin(), gold.end(), gold.begin(), std::negate<float>());
    EXPECT(migraphx::verify::verify_range(result_vector, gold));
}

TEST_CASE(nms_dyn_out_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_l,
        scores_l,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_dyn_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 3}, {6, 6}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 3}, {1, 1}, {6, 6}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0,
                                    0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {2, 1, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 1, 0, 3, 1, 0, 0, 1, 0, 5};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_dyn_boxes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 1}, {4, 20}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 1}, {1, 1}, {4, 20}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_dyn_classes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 1}, {6, 6}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 1}, {1, 3}, {6, 6}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{2});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                    0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 2, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_not_center_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                    1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    // set use_dyn_output back to false in operator map
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"use_dyn_output", false}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_transpose1_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 4, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), t_boxes_l);
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nms_transpose2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {4, 1, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t_boxes_l);
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(result, gold));
}

TEST_CASE(nonzero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> data = {
        1.0f, 1.3f, 0.0f, -1.2f, 0.0f, -100.f, 200.f, 0.0f, 0.1f, 0.2f, 0.0f, 0.5f};
    auto input = mm->add_literal(migraphx::literal(s, data));
    auto ret   = mm->add_instruction(migraphx::make_op("nonzero"), input);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                 1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(result_vector, gold));
}

TEST_CASE(not_test)
{
    // int32
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {4}};
        std::vector<float> data{0, 8, 1, -32};
        auto l1 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(migraphx::make_op("not"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<char> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<char> gold{1, 0, 0, 0};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    // bool
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::bool_type, {4}};
        std::vector<bool> data{false, false, true, true};
        auto l1 = mm->add_literal(migraphx::literal{s, {0, 0, 1, 1}});
        mm->add_instruction(migraphx::make_op("not"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<char> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<bool> gold(data.size());
        std::transform(
            data.begin(), data.end(), gold.begin(), [](bool n) -> bool { return not n; });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(not_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("not"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{0, 8, 1, -32};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<char> gold{1, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pad_test_asym)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 0, 3, 4, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pad_test_highest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::max()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::max();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pad_test_lowest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::lowest()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::lowest();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pad_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2}}, {2, 4, {2}}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1, 2, 3, 4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2}};
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pointwise_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1  = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2  = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    auto* pm = p.create_module("pointwise");
    auto x1  = pm->add_parameter("x1", {migraphx::shape::float_type});
    auto x2  = pm->add_parameter("x2", {migraphx::shape::float_type});
    pm->add_instruction(migraphx::make_op("add"), x1, x2);
    mm->add_instruction(migraphx::make_op("pointwise"), {l1, l2}, {pm});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pow_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {1, 2, 3};
    auto b                  = mm->add_literal(migraphx::literal{s, data});
    auto e                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("pow"), b, e);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::pow(n, n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(pow_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto b = mm->add_parameter("b", s);
    auto e = mm->add_parameter("e", s);
    mm->add_instruction(migraphx::make_op("pow"), b, e);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["b"] = migraphx::argument(input_fixed_shape0, data.data());
    params0["e"] = migraphx::argument(input_fixed_shape0, data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return std::pow(n, n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(prefix_scan_sum_1d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 10.0, 15.0, 21.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_dyn_1d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{5, 8}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {6}};
    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 10.0, 15.0, 21.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_2d)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_3d)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 2}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_exclusive)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.0, 1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 16.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", true}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_exclusive_reverse)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}, {"reverse", true}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{20.0, 18.0, 15.0, 11.0, 6.0, 0.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_negative_axis)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -3}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0,
                                2.0,
                                4.0,
                                6.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -2}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0,
                                1.0,
                                2.0,
                                3.0,
                                2.0,
                                4.0,
                                6.0,
                                3.0,
                                6.0,
                                9.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
        auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", -1}, {"exclusive", false}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0,
                                1.0,
                                3.0,
                                6.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prefix_scan_sum_reverse)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum",
                              {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
            l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{20.0, 19.0, 17.0, 14.0, 10.0, 9.0, 7.0, 4.0};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 2, 2}};
        auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
        auto l0    = mm->add_literal(input);
        mm->add_instruction(
            migraphx::make_op("prefix_scan_sum",
                              {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
            l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(prelu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto x     = mm->add_literal(migraphx::literal{s, {-1, 0, 2}});
    auto slope = mm->add_literal(migraphx::literal{s, {2, 1, 2}});
    mm->add_instruction(migraphx::make_op("prelu"), x, slope);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 0.0f, 2.0f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(prelu_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x     = mm->add_parameter("x", s);
    auto slope = mm->add_parameter("slope", s);
    mm->add_instruction(migraphx::make_op("prelu"), x, slope);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 2};
    std::vector<float> slope_data{2, 1, 2};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"]     = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["slope"] = migraphx::argument(input_fixed_shape0, slope_data.data());
    auto result      = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 0.0f, 2.0f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(quant_conv2d_padding_stride_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int32_t> s = {4521,
                              7014,
                              7830,
                              11952,
                              10515,
                              16734,
                              19737,
                              30906,
                              13161,
                              19542,
                              19494,
                              28800,
                              34707,
                              52590,
                              54729,
                              82746};
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_padding_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});
    mm->add_instruction(
        migraphx::make_op("quant_convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result            = p.eval({}).back();
    std::vector<int32_t> s = {
        4521,  6753,  7014,  4635,  6858,  10197, 10548, 6939,  7830,  11601, 11952, 7839,  5007,
        7383,  7590,  4953,  10515, 15987, 16734, 11277, 16821, 25506, 26586, 17874, 19737, 29826,
        30906, 20718, 13593, 20505, 21198, 14187, 13161, 19281, 19542, 12699, 18522, 27045, 27396,
        17739, 19494, 28449, 28800, 18639, 11919, 17319, 17526, 11289, 34707, 51843, 52590, 34893,
        51813, 77346, 78426, 52002, 54729, 81666, 82746, 54846, 36057, 53769, 54462, 36075};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(quant_conv2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::int8_type, {2, 3, 4, 4}};
    std::vector<int8_t> a(2 * 3 * 4 * 4);
    std::iota(a.begin(), a.end(), 0);
    auto al = mm->add_literal(migraphx::literal{a_shape, a});

    migraphx::shape c_shape{migraphx::shape::int8_type, {2, 3, 3, 3}};
    std::vector<int8_t> c(2 * 3 * 3 * 3);
    std::iota(c.begin(), c.end(), 0);
    auto cl = mm->add_literal(migraphx::literal{c_shape, c});

    mm->add_instruction(migraphx::make_op("quant_convolution"), al, cl);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<int32_t> s = {10197,
                              10548,
                              11601,
                              11952,
                              25506,
                              26586,
                              29826,
                              30906,
                              27045,
                              27396,
                              28449,
                              28800,
                              77346,
                              78426,
                              81666,
                              82746};

    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(quantizelinear)
{
    {
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 4, 4, 4, 6, 6, 6, 2, 2, 2, 4, 4, 4, 6, 6, 6};
        migraphx::shape zs{migraphx::shape::int8_type, {2, 3, 3}};
        std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto create_program     = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            auto z   = mm->add_literal(zs, zv);
            mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::make_target("ref"));
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{
            -128, 127, 65, -128, 1, 1, -1, 100, 92, -128, 127, 65, -128, 1, 1, -1, 100, 92};
        EXPECT(results_vector == gold);
    }

    {
        migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> xv = {
            -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
        migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
        std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        auto create_program   = [&]() {
            migraphx::program p;
            auto* mm = p.get_main_module();
            auto x   = mm->add_literal(xs, xv);
            auto s   = mm->add_literal(ss, sv);
            mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
            return p;
        };

        migraphx::program p1 = create_program();
        p1.compile(migraphx::make_target("ref"));
        auto result = p1.eval({}).back();
        std::vector<float> results_vector(18);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 255, 65, 0, 2, 2, 0, 255, 255, 0, 255, 65, 0, 2, 2, 0, 255, 255};
        EXPECT(results_vector == gold);
    }
}

TEST_CASE(recip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::double_type, {3}};
    std::vector<float> data{-0.5f, 0.1f, 0.5f};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("recip"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 10.0f, 2.0f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(recip_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("recip"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-0.5f, 0.1f, 0.5f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2.0f, 10.0f, 2.0f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(reduce_max_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{9, 10, 11, 12};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_max_dynamic_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2}}, {3, 5, {3}}}};
    auto input         = mm->add_parameter("X", s);
    auto reduce_max_op = migraphx::make_op("reduce_max", {{"axes", {0}}});
    mm->add_instruction(reduce_max_op, input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 5}};
    std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    params["X"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {6, 7, 8, 9, 10};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(reduce_max_axis01)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{11, 12};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_max_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{10, 12};
    EXPECT(results_vector == gold);
}

template <typename T, bool dynamic = false>
void reduce_op_axes_input_test(const std::string op_name,
                               std::vector<int64_t> axes,
                               std::vector<T> gold)
{
    migraphx::program p;
    auto* mm               = p.get_main_module();
    auto data_static_shape = migraphx::shape{migraphx::shape::get_type<T>{}, {3, 2, 2}};
    auto data_shape        = dynamic ? migraphx::shape{migraphx::shape::get_type<T>{},
                                                {{2, 5, {}}, {2, 3, {}}, {2, 3, {}}}}
                                     : data_static_shape;
    auto data_param        = mm->add_parameter("data", data_shape);
    auto axes_shape        = migraphx::shape{migraphx::shape::int64_type, {axes.size()}};
    auto axes_param        = mm->add_parameter("axes", axes_shape);
    mm->add_instruction(migraphx::make_op(op_name, {}), data_param, axes_param);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    T data[12]     = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    params["data"] = migraphx::argument(data_static_shape, data);
    params["axes"] = migraphx::argument(axes_shape, axes.data());
    auto result    = p.eval(params).back();

    std::vector<T> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_max_axes_input_axis0)
{
    reduce_op_axes_input_test<float>("reduce_max", {0}, {9, 10, 11, 12});
}

TEST_CASE(reduce_max_axes_input_axes01)
{
    reduce_op_axes_input_test<float>("reduce_max", {0, 1}, {11, 12});
}

TEST_CASE(reduce_max_axes_input_axes02)
{
    reduce_op_axes_input_test<float>("reduce_max", {0, 2}, {10, 12});
}

TEST_CASE(reduce_max_dynamic_data_axes_input_axis0)
{
    reduce_op_axes_input_test<float, true>("reduce_max", {0}, {9, 10, 11, 12});
}

TEST_CASE(reduce_mean_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{5.5, 7.5};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2, 3, 6, 7, 10, 11};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.5f, 6.5f, 10.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_int)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold{2, 6, 10};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_mean_axes_input_axes02)
{
    reduce_op_axes_input_test<float>("reduce_mean", {0, 2}, {5.5f, 7.5f});
}

TEST_CASE(reduce_mean_axes_input_axis1)
{
    reduce_op_axes_input_test<float>("reduce_mean", {1}, {2, 3, 6, 7, 10, 11});
}

TEST_CASE(reduce_mean_axes_input_axes12)
{
    reduce_op_axes_input_test<float>("reduce_mean", {1, 2}, {2.5f, 6.5f, 10.5f});
}

TEST_CASE(reduce_mean_axes_input_axis2)
{
    reduce_op_axes_input_test<float>("reduce_mean", {2}, {1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f});
}

TEST_CASE(reduce_mean_int_axes_input_axes12)
{
    reduce_op_axes_input_test<int32_t>("reduce_mean", {1, 2}, {2, 6, 10});
}

TEST_CASE(reduce_min_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 3};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 5, 6, 9, 10};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 5, 9};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_min_axes_input_axes02)
{
    reduce_op_axes_input_test<float>("reduce_min", {0, 2}, {1, 3});
}

TEST_CASE(reduce_min_axes_input_axis1)
{
    reduce_op_axes_input_test<float>("reduce_min", {1}, {1, 2, 5, 6, 9, 10});
}

TEST_CASE(reduce_min_axes_input_axes12)
{
    reduce_op_axes_input_test<float>("reduce_min", {1, 2}, {1, 5, 9});
}

TEST_CASE(reduce_prod_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_prod", {{"axes", {0}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{6, 18, 12, 18};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{15, 18, 21, 24};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis02)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{33, 45};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4, 6, 12, 14, 20, 22};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis12)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{10, 26, 42};
    EXPECT(results_vector == gold);
}

TEST_CASE(reduce_sum_axis2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{3, 7, 11, 15, 19, 23};
    EXPECT(results_vector == gold);
}

TEST_CASE(relu_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {-1.f, 0.f, 1.f}});
    mm->add_instruction(migraphx::make_op("relu"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.f, 0.f, 1.f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(relu_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("relu"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1.f, 0.f, 1.f};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.f, 0.f, 1.f};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(reshape_test0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {8, 3, 1, 1};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, data));
}

TEST_CASE(reshape_test1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {1, 3, 4, 2};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, data));
}

TEST_CASE(reshape_test2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {1, 2, 3, 4};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, data));
}

TEST_CASE(reshape_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    std::vector<int64_t> new_shape = {0, 8, 3, 1};
    auto input                     = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(48);
    std::iota(data.begin(), data.end(), -3);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 24, 1, 1}};
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, data));
}

TEST_CASE(reverse_test_axis0)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::swap_ranges(target_data.begin(), target_data.begin() + 16, target_data.begin() + 16);
    EXPECT(migraphx::verify::verify_range(results_vector, target_data));
}

TEST_CASE(reverse_test_axis1)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::reverse(target_data.begin(), target_data.begin() + 16);
    std::reverse(target_data.end() - 16, target_data.end());
    EXPECT(migraphx::verify::verify_range(results_vector, target_data));
}

TEST_CASE(reverse_test_axis10)
{
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> data(32);
    std::iota(data.begin(), data.end(), 1);
    migraphx::program p;
    auto* mm              = p.get_main_module();
    auto l                = mm->add_literal(migraphx::literal{in_shape, data});
    std::vector<int> axes = {1, 0};
    mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> target_data = data;
    std::reverse(target_data.begin(), target_data.begin() + 16);
    std::reverse(target_data.end() - 16, target_data.end());
    std::swap_ranges(target_data.begin(), target_data.begin() + 16, target_data.begin() + 16);
    EXPECT(migraphx::verify::verify_range(results_vector, target_data));
}

TEST_CASE(roialign_out_of_bound_test)
{
    auto create_program = [](const std::string& trans_mode = "half_pixel") {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
        std::vector<float> x_vec = {
            0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
            0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
            0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
            0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
            0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
            0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
            0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
            0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
            0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
            0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

        migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
        std::vector<float> roi_vec = {0, 0, 9.99, 9.99, 0, 5, 4, 9, 5, 5, 9.9, 9.9};

        migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> ind_vec = {0, 0, 0};

        auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
        auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r =
            mm->add_instruction(migraphx::make_op("roialign",
                                                  {{"coordinate_transformation_mode", trans_mode},
                                                   {"spatial_scale", 5.0},
                                                   {"output_height", 1},
                                                   {"output_width", 1},
                                                   {"sampling_ratio", 1}}),
                                x,
                                roi,
                                ind);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program("output_half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0.0f, 0.0f, 0.0f};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(roialign_test)
{
    auto create_program = [](const std::string& trans_mode = "half_pixel",
                             const migraphx::op::pooling_mode pooling_mode =
                                 migraphx::op::pooling_mode::average,
                             int64_t sampling_ratio = 2) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
        std::vector<float> x_vec = {
            0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
            0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
            0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
            0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
            0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
            0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
            0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
            0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
            0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
            0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

        migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
        std::vector<float> roi_vec = {0, 0, 9, 9, 0, 5, 4, 9, 5, 5, 9, 9};

        migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> ind_vec = {0, 0, 0};

        auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
        auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r =
            mm->add_instruction(migraphx::make_op("roialign",
                                                  {{"coordinate_transformation_mode", trans_mode},
                                                   {"spatial_scale", 1.0},
                                                   {"output_height", 5},
                                                   {"output_width", 5},
                                                   {"sampling_ratio", sampling_ratio},
                                                   {"mode", pooling_mode}}),
                                x,
                                roi,
                                ind);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program();
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.466421425, 0.446552634, 0.340521216, 0.568848491, 0.606780827, 0.371379346,
            0.429571986, 0.383519977, 0.556241512, 0.351050019, 0.27680251,  0.488286227,
            0.522200167, 0.552770197, 0.417057365, 0.471240699, 0.4844096,   0.690457463,
            0.492039412, 0.877398551, 0.623889625, 0.712461948, 0.628926516, 0.335504025,
            0.349469036, 0.302179992, 0.43046391,  0.469585985, 0.39774403,  0.542259991,
            0.365552008, 0.704923987, 0.516481996, 0.317131996, 0.701444089, 0.291239977,
            0.505897999, 0.647610962, 0.623489916, 0.829879999, 0.591567993, 0.738860011,
            0.704825997, 0.837148011, 0.889315963, 0.622680008, 0.615276039, 0.709713995,
            0.615356028, 0.458524048, 0.238451958, 0.337952018, 0.371693879, 0.609999895,
            0.760059953, 0.376724035, 0.378532052, 0.71468991,  0.924308002, 0.972783983,
            0.574903965, 0.582623959, 0.570936024, 0.761904061, 0.876998067, 0.535508037,
            0.256580025, 0.214098021, 0.279604018, 0.360000014, 0.436488032, 0.350427985,
            0.288755983, 0.366139978, 0.234920025};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        auto p = create_program("output_half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.517783, 0.343411, 0.322905, 0.447362, 0.634375, 0.40308,  0.536647, 0.442791,
            0.486144, 0.402313, 0.251194, 0.400154, 0.515524, 0.695369, 0.346537, 0.33504,
            0.460099, 0.588069, 0.343863, 0.684932, 0.49319,  0.714058, 0.821744, 0.471935,
            0.403946, 0.306955, 0.218678, 0.33369,  0.488001, 0.486962, 0.18709,  0.49142,
            0.55611,  0.419167, 0.368608, 0.143278, 0.460835, 0.597125, 0.53096,  0.498207,
            0.278818, 0.438569, 0.6022,   0.700038, 0.752436, 0.577385, 0.702383, 0.725097,
            0.733754, 0.816304, 0.23933,  0.407514, 0.337893, 0.252521, 0.474335, 0.367075,
            0.270168, 0.41051,  0.64189,  0.830777, 0.55564,  0.454295, 0.55645,  0.75015,
            0.929997, 0.66257,  0.561664, 0.481275, 0.495449, 0.666306, 0.663573, 0.372107,
            0.205603, 0.192776, 0.247849};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        auto p = create_program("output_half_pixel", migraphx::op::pooling_mode::max, 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {
            0.819145, 0.373103, 0.258302,  0.515419, 0.726104, 0.540536, 0.545512,  0.38511,
            0.376545, 0.274635, 0.22341,   0.184511, 0.230843, 0.404869, 0.29546,   0.540409,
            0.265838, 0.409324, 0.213915,  0.708654, 0.687264, 0.580821, 0.461283,  0.462879,
            0.709632, 0.27873,  0.083619,  0.22428,  0.313992, 0.410508, 0.0929099, 0.415373,
            0.296695, 0.231574, 0.136836,  0.0683,   0.296695, 0.211925, 0.245385,  0.28053,
            0.17091,  0.179879, 0.245385,  0.343539, 0.392742, 0.51273,  0.536193,  0.382995,
            0.422793, 0.761886, 0.0839429, 0.276444, 0.19746,  0.126117, 0.378351,  0.254646,
            0.092148, 0.272825, 0.381955,  0.626599, 0.251325, 0.244475, 0.194875,  0.272825,
            0.44757,  0.351855, 0.342265,  0.244475, 0.274841, 0.553644, 0.607176,  0.202392,
            0.07425,  0.066087, 0.126279};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(round_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {9}};
    auto l =
        mm->add_literal(migraphx::literal{s, {1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0}});
    mm->add_instruction(migraphx::make_op("round"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 2.0, 2.0, -1.0, -2.0, -2.0, 0.0, 2.0, -2.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(round_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{4, 10};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("round"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1.1, 1.5, 1.6, -1.1, -1.5, -1.6, 0.0, 2.0, -2.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {9}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 2.0, 2.0, -1.0, -2.0, -2.0, 0.0, 2.0, -2.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(rsqrt_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = mm->add_literal(migraphx::literal{s, {4.0, 16.0, 64.0}});
    mm->add_instruction(migraphx::make_op("rsqrt"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.5, 0.25, 0.125};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(rsqrt_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("rsqrt"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{4.0, 16.0, 64.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.5, 0.25, 0.125};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

// reduction_mode: "scatter_none", "scatter_add", "scatter_mul"
migraphx::program create_scatter_program(const std::string& reduction_mode, int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 0.0f);

    migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> vi = {1, 0, 2, 0, 2, 1};

    migraphx::shape su{migraphx::shape::float_type, {2, 3}};
    std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    // scatter_none, formerly the scatter op
    auto r = mm->add_instruction(migraphx::make_op(reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}

TEST_CASE(scatter_ax0_test)
{
    // this tests what used to be the only scatter op, now changed to 3 sub-ops
    // which have their own test case
    {
        migraphx::program p = create_scatter_program("scatter_none", 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(scatter_ax_neg_test)
{
    {
        migraphx::program p = create_scatter_program("scatter_none", -2);

        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(scatter_ax1_test)
{
    {
        migraphx::program p = create_scatter_program("scatter_none", 1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {1.1, 1.0, 1.2, 2.0, 2.2, 2.1, 0.0, 0.0, 0.0};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

// similar to create_scatter_program but with different tensor values
// reduction_mode: "scatter_none", "scatter_add", "scatter_mul"
migraphx::program create_scatter_program2(const std::string& reduction_mode, int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {1, 5}};
    std::vector<float> vd({1., 2., 3., 4., 5.});

    migraphx::shape si{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> vi = {1, 3};

    migraphx::shape su{migraphx::shape::float_type, {1, 2}};
    std::vector<float> vu = {1.1, 2.1};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(migraphx::make_op(reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}
TEST_CASE(scatter_reduction1_test)
{
    {
        // Test sub-ops for the three reduction values scatter_none, scatter_add, scatter_mul
        migraphx::program p = create_scatter_program2("scatter_none", 1);

        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_none = {1.0, 1.1, 3.0, 2.1, 5.0};
        EXPECT(migraphx::verify::verify_range(results_vector, gold_none));
    }
}

TEST_CASE(scatter_reduction2_test)
{
    {
        migraphx::program p = create_scatter_program2("scatter_mul", 1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_mul = {1.0, 2.2, 3.0, 8.4, 5.0};

        EXPECT(migraphx::verify::verify_range(results_vector, gold_mul));
    }
}
TEST_CASE(scatter_reduction3_test)
{
    {
        migraphx::program p = create_scatter_program2("scatter_add", 1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_add = {1.0, 3.1, 3.0, 6.1, 5.0};

        EXPECT(migraphx::verify::verify_range(results_vector, gold_add));
    }
}

TEST_CASE(scatter_reduction_3x3_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
        std::vector<float> vd(sd.elements(), 3.0f);

        migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
        std::vector<int> vi = {1, 0, 2, 0, 2, 1};

        migraphx::shape su{migraphx::shape::float_type, {2, 3}};
        std::vector<float> vu = {1.0, 1.1, 1.2, 7.0, 7.1, 7.2};

        auto ld = mm->add_literal(migraphx::literal{sd, vd});
        auto li = mm->add_literal(migraphx::literal{si, vi});
        auto lu = mm->add_literal(migraphx::literal{su, vu});
        auto r  = mm->add_instruction(migraphx::make_op("scatter_add", {{"axis", 1}}), ld, li, lu);
        mm->add_return({r});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_a2 = {4.1, 4.0, 4.2, 10.0, 10.2, 10.1, 3.0, 3.0, 3.0};

        EXPECT(migraphx::verify::verify_range(results_vector, gold_a2));
    }
}

// create a test scatter program with a 3x3 tensor;
//  su and si are transposed from previous case
migraphx::program create_scatter_program_3x3(const std::string& reduction_mode, int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 3.0f);

    migraphx::shape si{migraphx::shape::int32_type, {3, 2}};
    std::vector<int> vi = {1, 0, 0, 2, 2, 1};

    migraphx::shape su{migraphx::shape::float_type, {3, 2}};
    std::vector<float> vu = {1.0, 7.0, 1.1, 7.1, 1.2, 7.2};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(migraphx::make_op(reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}

TEST_CASE(scatter_reduction_3x3_xpose1_test)
{
    // test on vertical (0) axis. su and si are transposed from previous case
    {
        migraphx::program p = create_scatter_program_3x3("scatter_none", 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_none2 = {1.1, 7.0, 3.0, 1.0, 7.2, 3.0, 1.2, 7.1, 3.0};
        EXPECT(migraphx::verify::verify_range(results_vector, gold_none2));
    }
}

TEST_CASE(scatter_reduction_3x3_xpose2_test)
{
    // test on vertical (0) axis.
    {
        migraphx::program p = create_scatter_program_3x3("scatter_add", 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_a3 = {4.1, 10.0, 3.0, 4.0, 10.2, 3.0, 4.2, 10.1, 3.0};

        EXPECT(migraphx::verify::verify_range(results_vector, gold_a3));
    }
}

TEST_CASE(scatter_reduction_3x3_xpose3_test)
{
    {
        migraphx::program p = create_scatter_program_3x3("scatter_mul", 0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold_mul2 = {3.3, 21.0, 3.0, 3.0, 21.6, 3.0, 3.6, 21.3, 3.0};

        EXPECT(migraphx::verify::verify_range(results_vector, gold_mul2));
    }
}

TEST_CASE(scatternd_shapes_test)
{
    {
        // broadcasted input
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {4}};

        std::vector<int64_t> ind_vec{4, 3, 1, 7};
        std::vector<float> upd_vec{9, 10, 11, 12};

        auto data    = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}),
                                        mm->add_literal(migraphx::literal{0.0f}));
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{0, 11, 0, 10, 9, 0, 0, 12};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // non-standard shape input
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {2, 2}};
        migraphx::shape is{itype, {2, 2}};
        migraphx::shape us{dtype, {2}};

        std::vector<float> data_vec{1, 2, 3, 4};
        std::vector<int64_t> ind_vec{0, 0, 0, 1};
        std::vector<float> upd_vec{5, 6};

        auto data = mm->add_literal(migraphx::literal{ds, data_vec});
        auto td =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), data);
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), td, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{5, 6, 2, 4};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // non-standard updates shape
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {2, 2, 2}};
        migraphx::shape is{itype, {2, 1, 3}};
        migraphx::shape us{dtype, {1, 2}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{0, 0, 0, 1, 1, 1};
        std::vector<float> upd_vec{9, 10};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto tu =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), updates);
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, tu);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{9, 2, 3, 4, 5, 6, 7, 10};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(scatternd_test)
{
    {
        // r=1, q=2, k=1
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {4}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};
        std::vector<float> upd_vec{9, 10, 11, 12};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1, 11, 3, 10, 9, 6, 7, 12};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // r=2, q=2, k=2
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {2, 2}};
        migraphx::shape is{itype, {2, 2}};
        migraphx::shape us{dtype, {2}};

        std::vector<float> data_vec{1, 2, 3, 4};
        std::vector<int64_t> ind_vec{0, 0, 0, 1};
        std::vector<float> upd_vec{5, 6};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{5, 6, 3, 4};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // r=3, q=3, k=3
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {2, 2, 2}};
        migraphx::shape is{itype, {2, 1, 3}};
        migraphx::shape us{dtype, {2, 1}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{0, 0, 0, 1, 1, 1};
        std::vector<float> upd_vec{9, 10};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{9, 2, 3, 4, 5, 6, 7, 10};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // r=3, q=2, k=1
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {4, 4, 4}};
        migraphx::shape is{itype, {2, 1}};
        migraphx::shape us{dtype, {2, 4, 4}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                    1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                    8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                    8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{0, 2};
        std::vector<float> upd_vec{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 1, 2, 3, 4, 5, 6,
                                7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                4, 4, 4, 4, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // r=5, q=1, k=1
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {2, 2, 2, 2, 2}};
        migraphx::shape is{itype, {1}};
        migraphx::shape us{dtype, {2, 2, 2, 2}};

        std::vector<float> data_vec(32, 1);
        std::vector<int64_t> ind_vec{1};
        std::vector<float> upd_vec(16, 0);

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold(32, 0);
        std::copy(data_vec.begin(), data_vec.begin() + 16, gold.begin());

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(scatternd_reduction_test)
{
    {
        // reduction = add
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {8, 1}};
        migraphx::shape us{dtype, {8}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{4, 3, 1, 7, 4, 3, 1, 7};
        std::vector<float> upd_vec{9, 10, 11, 12, -8, -9, -10, -11};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_add"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1, 3, 3, 5, 6, 6, 7, 9};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }

    {
        // reduction = mul
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {4}};

        std::vector<float> data_vec{1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int64_t> ind_vec{4, 3, 1, 7};
        std::vector<float> upd_vec{9, 10, 11, 12};

        auto data    = mm->add_literal(migraphx::literal{ds, data_vec});
        auto indices = mm->add_literal(migraphx::literal{is, ind_vec});
        auto updates = mm->add_literal(migraphx::literal{us, upd_vec});
        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_mul"), data, indices, updates);
        mm->add_return({scatternd});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold{1, 22, 3, 40, 45, 6, 7, 96};

        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(select_module_add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        submod->add_return({add_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2, 14, 5, 10, 5, 14, 14, 2};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test0)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test1)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4, -4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {4, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4, -5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(select_module_not_found_error)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8,  8,  -4, -4, 8,
                                  -1, 4, -1, 8, 8,  -4, -1, 8,  8,  -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {5, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

TEST_CASE(scatternd_reduction_dyn_test)
{
    // reduction = add, with dynamic input shapes
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto dtype = migraphx::shape::float_type;
    auto itype = migraphx::shape::int64_type;
    migraphx::shape::dynamic_dimension dd{3, 6};
    migraphx::shape ds{migraphx::shape::float_type, {dd, dd, dd}};
    migraphx::shape is{itype, {2, 1}};
    migraphx::shape us{dtype, {{2, 2}, dd, dd}};

    auto xdata    = mm->add_parameter("X", ds);
    auto xindex   = mm->add_parameter("I", is);
    auto xupdates = mm->add_parameter("U", us);

    auto scatternd_add_op = migraphx::make_op("scatternd_add");
    auto scatternd        = mm->add_instruction(scatternd_add_op, xdata, xindex, xupdates);
    mm->add_return({scatternd});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4, 4, 4}}; // data
    std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6,
                                  7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4,
                                  5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> input_index{0, 2};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {2, 4, 4}}; // updates
    std::vector<float> input_updates{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                     1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

    params["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    params["I"] = migraphx::argument(is, input_index.data());
    params["U"] = migraphx::argument(input_fixed_shape1, input_updates.data());

    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{6, 7, 8, 9, 11, 12, 13, 14, 15, 14, 13, 12, 12, 11, 10, 9,
                            1, 2, 3, 4, 5,  6,  7,  8,  8,  7,  6,  5,  4,  3,  2,  1,
                            9, 8, 7, 6, 6,  5,  4,  3,  4,  5,  6,  7,  9,  10, 11, 12,
                            8, 7, 6, 5, 4,  3,  2,  1,  1,  2,  3,  4,  5,  6,  7,  8};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sigmoid_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l = mm->add_literal(migraphx::literal{s, {-1, 2, -3, 4}});
    mm->add_instruction(migraphx::make_op("sigmoid"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{sigmoid(-1), sigmoid(2), sigmoid(-3), sigmoid(4)};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sigmoid_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {2, 2}}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("sigmoid"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 2, -3, 4};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 2}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{sigmoid(-1), sigmoid(2), sigmoid(-3), sigmoid(4)};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sign_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto l = mm->add_literal(
        migraphx::literal{s, {1.02481645, 0.85643062, -0.03404123, -0.92791926, 0.0}});
    mm->add_instruction(migraphx::make_op("sign"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 1.0, -1.0, -1.0, 0.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sign_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("sign"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1.02481645, 0.85643062, -0.03404123, -0.92791926, 0.0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {5}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.0, 1.0, -1.0, -1.0, 0.0};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sin_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data = {-1, 0, 1};
    auto l                  = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sin"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sin_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("sin"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {-1, 0, 1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sinh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data{-1.0, 2.0, -3.0, 4.0};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sinh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sinh_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {2, 4}}};
    auto input = mm->add_parameter("X", s);
    std::vector<float> input_data{-1.0, 2.0, -3.0, 4.0};
    mm->add_instruction(migraphx::make_op("sinh"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sinhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(slice_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}), l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_output_shapes().back() == s2);
        p.compile(migraphx::make_target("ref"));
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({}).back();
        std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<int> data(2 * 2 * 3);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s, data});
        mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
            l0);
        migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
        EXPECT(p.get_output_shapes().back() == s2);
        p.compile(migraphx::make_target("ref"));
        migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
        auto result           = p.eval({}).back();
        std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
        std::vector<int> results_vector(2 * 2 * 2);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
        EXPECT(result.get_shape() == sresult);
    }
}

TEST_CASE(slice_var_inputs_static0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), l0, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {1};
    std::vector<int32_t> end_data   = {3};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), l0, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {-2};
    std::vector<int32_t> end_data   = {2831};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::float_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {3}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    auto axes   = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice"), l0, starts, ends, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> start_data = {0, 0, 0};
    std::vector<int64_t> end_data   = {2, 2, 2};
    std::vector<int64_t> axes_data  = {0, 1, 2};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    params["axes"]                  = migraphx::argument(s1, axes_data.data());
    auto result                     = p.eval(params).back();
    std::vector<float> gold         = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<float> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), input, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> start_data = {1};
    std::vector<int> end_data   = {3};
    params["input"]             = migraphx::argument(s2, input_data.data());
    params["starts"]            = migraphx::argument(s1, start_data.data());
    params["ends"]              = migraphx::argument(s1, end_data.data());
    auto result                 = p.eval(params).back();
    std::vector<int> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(slice_dyn_test0)
{
    // Slice a single dynamic dimension. ax1 slice limits are smaller than min; ax2 "ends" is
    // too large
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 3}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1, 2}}, {"starts", {0, 1}}, {"ends", {1, 6}}}), x);
    migraphx::shape s2{migraphx::shape::int32_type, {{2, 3}, {1, 1}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));

    //  the strides of sresult are those of the original shape, not
    // reduced to sliced size.
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 1, 2}, {6, 3, 1}};
    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 2, 7, 8};
    std::vector<int> results_vector(2 * 1 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_dyn_test1)
{
    // Slice all three dynamic dimensions
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
        x);

    migraphx::shape s2{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(softmax_simple_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {0.25, 0.75};
    std::vector<float> s = {0.377541, 0.622459};
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(softmax_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -5.61869681e-01, 9.07827199e-01,  1.29255986e+00,  3.18533443e-02,  -1.22183852e-03,
        -2.83830553e-01, -1.03245842e+00, -9.28322077e-01, -8.82696748e-01, 1.11327164e-01,
        -9.20038462e-01, 8.47388089e-01,  2.51734018e-01,  1.50563884e+00,  2.23056650e+00,
        -6.17576987e-02, -1.00264274e-01, -6.10369384e-01, 1.17537189e+00,  -2.51560897e-01,
        -8.50333512e-01, -8.03578615e-01, -6.51194930e-01, -2.58137047e-01, 4.65528190e-01,
        3.23284641e-02,  -1.54700470e+00, 1.38096774e+00,  5.39869189e-01,  -7.56884992e-01,
        1.81503093e+00,  -2.11269641e+00, 1.92466557e+00,  1.77230799e+00,  2.21660900e+00,
        1.56777036e+00,  -2.08995026e-03, 3.50566894e-01,  -1.15042710e+00, -1.18577778e+00,
        8.90633047e-01,  -6.63949102e-02, 1.44661188e+00,  1.59215283e+00,  -2.56262213e-01,
        9.39079225e-01,  4.07298543e-02,  3.86590779e-01,  6.09607756e-01,  8.22331488e-01,
        -2.82126725e-01, -9.49052632e-01, -4.24012303e-01, -5.32990396e-01, -3.18386006e+00,
        3.27092171e-01,  -1.33315325e+00, 3.62459183e-01,  3.74710828e-01,  -1.30302286e+00,
        1.79680198e-01,  -4.51832324e-01, 4.34282750e-01,  -7.09520102e-01, 6.20333970e-01,
        -1.28712380e+00, 2.04130828e-01,  -7.70607769e-01, 1.61889160e+00,  -1.50951004e+00,
        -4.10505563e-01, -3.56566496e-02, -1.29747534e+00, -1.49967879e-01, 7.77626812e-01,
        -8.28408226e-02, 2.73412596e-02,  5.79780899e-03,  9.87900198e-02,  -7.95276761e-01,
        -1.38536084e+00, -6.63573861e-01, 3.89783204e-01,  -1.30670881e+00, -7.62425125e-01,
        -4.04883057e-01, 6.24344349e-01,  3.68128955e-01,  -1.01577950e+00, -3.06715906e-01,
        5.67961395e-01,  2.98198581e-01,  -1.63613629e+00, -3.75131965e-01, -6.75393403e-01,
        2.59172034e+00,  6.75538957e-01,  9.07939598e-02,  1.92257717e-01,  -1.21592450e+00,
        -2.73682117e-01, 1.25232983e+00,  -1.39969170e+00, -1.91483587e-01, 2.57732719e-01,
        3.10056299e-01,  1.41833842e+00,  -1.81386679e-01, 3.92868072e-01,  -8.14771175e-01,
        2.02392387e+00,  -9.42091495e-02, -3.77683818e-01, 2.05638766e+00,  2.93796062e-01,
        -6.02131486e-01, 2.70461679e-01,  -8.92358482e-01, 1.04388881e+00,  2.66154885e-01};

    std::vector<float> s = {
        0.30191708, 0.59879845, 0.50029165, 0.24915339, 0.36823985, 0.13190967, 0.0349741,
        0.18750034, 0.21905553, 0.27000085, 0.0547399,  0.56318235, 0.47422904, 0.78964758,
        0.91381913, 0.44601166, 0.47902739, 0.13120073, 0.4449684,  0.18766427, 0.15753111,
        0.07844277, 0.05120674, 0.36648798, 0.14637007, 0.13152322, 0.01560997, 0.29065287,
        0.49196178, 0.10550152, 0.81890774, 0.06369215, 0.62972021, 0.74931765, 0.67285055,
        0.35034987, 0.28612873, 0.31931475, 0.04220394, 0.16093165, 0.22390974, 0.11915915,
        0.3115395,  0.35899726, 0.22190949, 0.57518375, 0.13888834, 0.7753762,  0.4642328,
        0.57055861, 0.21954368, 0.34515455, 0.09486015, 0.40631217, 0.01842281, 0.48770609,
        0.06652815, 0.36023033, 0.42343026, 0.24226256, 0.17348589, 0.44066274, 0.6865865,
        0.17296699, 0.46923906, 0.06921105, 0.3570261,  0.4125829,  0.73165393, 0.15302512,
        0.29499072, 0.33932695, 0.30852377, 0.40762195, 0.40170741, 0.36259529, 0.60848355,
        0.42618036, 0.31721094, 0.02960522, 0.28256637, 0.24389413, 0.2725659,  0.10663581,
        0.27622163, 0.28264219, 0.53652936, 0.09476089, 0.40890986, 0.34848392, 0.32572666,
        0.53076893, 0.11529481, 0.29117745, 0.14625968, 0.8756339,  0.49818122, 0.10656087,
        0.1813329,  0.17664003, 0.21410346, 0.80408043, 0.02315119, 0.27155462, 0.32804728,
        0.13268511, 0.61795473, 0.49703068, 0.41696799, 0.10175809, 0.71028161, 0.29929739,
        0.17377149, 0.76075399, 0.20071237, 0.32632929, 0.36892858, 0.09416146, 0.26656723,
        0.42914796};

    migraphx::shape a_shape{migraphx::shape::float_type, {5, 3, 4, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(120);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(softmax_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type,
                            {{1, 10}, {1, 3, {3}}, {4, 4}, {2, 2, {2}}}};
    auto al = mm->add_parameter("a", a_shape);
    mm->add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), al);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {
        -5.61869681e-01, 9.07827199e-01,  1.29255986e+00,  3.18533443e-02,  -1.22183852e-03,
        -2.83830553e-01, -1.03245842e+00, -9.28322077e-01, -8.82696748e-01, 1.11327164e-01,
        -9.20038462e-01, 8.47388089e-01,  2.51734018e-01,  1.50563884e+00,  2.23056650e+00,
        -6.17576987e-02, -1.00264274e-01, -6.10369384e-01, 1.17537189e+00,  -2.51560897e-01,
        -8.50333512e-01, -8.03578615e-01, -6.51194930e-01, -2.58137047e-01, 4.65528190e-01,
        3.23284641e-02,  -1.54700470e+00, 1.38096774e+00,  5.39869189e-01,  -7.56884992e-01,
        1.81503093e+00,  -2.11269641e+00, 1.92466557e+00,  1.77230799e+00,  2.21660900e+00,
        1.56777036e+00,  -2.08995026e-03, 3.50566894e-01,  -1.15042710e+00, -1.18577778e+00,
        8.90633047e-01,  -6.63949102e-02, 1.44661188e+00,  1.59215283e+00,  -2.56262213e-01,
        9.39079225e-01,  4.07298543e-02,  3.86590779e-01,  6.09607756e-01,  8.22331488e-01,
        -2.82126725e-01, -9.49052632e-01, -4.24012303e-01, -5.32990396e-01, -3.18386006e+00,
        3.27092171e-01,  -1.33315325e+00, 3.62459183e-01,  3.74710828e-01,  -1.30302286e+00,
        1.79680198e-01,  -4.51832324e-01, 4.34282750e-01,  -7.09520102e-01, 6.20333970e-01,
        -1.28712380e+00, 2.04130828e-01,  -7.70607769e-01, 1.61889160e+00,  -1.50951004e+00,
        -4.10505563e-01, -3.56566496e-02, -1.29747534e+00, -1.49967879e-01, 7.77626812e-01,
        -8.28408226e-02, 2.73412596e-02,  5.79780899e-03,  9.87900198e-02,  -7.95276761e-01,
        -1.38536084e+00, -6.63573861e-01, 3.89783204e-01,  -1.30670881e+00, -7.62425125e-01,
        -4.04883057e-01, 6.24344349e-01,  3.68128955e-01,  -1.01577950e+00, -3.06715906e-01,
        5.67961395e-01,  2.98198581e-01,  -1.63613629e+00, -3.75131965e-01, -6.75393403e-01,
        2.59172034e+00,  6.75538957e-01,  9.07939598e-02,  1.92257717e-01,  -1.21592450e+00,
        -2.73682117e-01, 1.25232983e+00,  -1.39969170e+00, -1.91483587e-01, 2.57732719e-01,
        3.10056299e-01,  1.41833842e+00,  -1.81386679e-01, 3.92868072e-01,  -8.14771175e-01,
        2.02392387e+00,  -9.42091495e-02, -3.77683818e-01, 2.05638766e+00,  2.93796062e-01,
        -6.02131486e-01, 2.70461679e-01,  -8.92358482e-01, 1.04388881e+00,  2.66154885e-01};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {5, 3, 4, 2}};
    params["a"] = migraphx::argument(input_fixed_shape, a.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(120);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> s = {
        0.30191708, 0.59879845, 0.50029165, 0.24915339, 0.36823985, 0.13190967, 0.0349741,
        0.18750034, 0.21905553, 0.27000085, 0.0547399,  0.56318235, 0.47422904, 0.78964758,
        0.91381913, 0.44601166, 0.47902739, 0.13120073, 0.4449684,  0.18766427, 0.15753111,
        0.07844277, 0.05120674, 0.36648798, 0.14637007, 0.13152322, 0.01560997, 0.29065287,
        0.49196178, 0.10550152, 0.81890774, 0.06369215, 0.62972021, 0.74931765, 0.67285055,
        0.35034987, 0.28612873, 0.31931475, 0.04220394, 0.16093165, 0.22390974, 0.11915915,
        0.3115395,  0.35899726, 0.22190949, 0.57518375, 0.13888834, 0.7753762,  0.4642328,
        0.57055861, 0.21954368, 0.34515455, 0.09486015, 0.40631217, 0.01842281, 0.48770609,
        0.06652815, 0.36023033, 0.42343026, 0.24226256, 0.17348589, 0.44066274, 0.6865865,
        0.17296699, 0.46923906, 0.06921105, 0.3570261,  0.4125829,  0.73165393, 0.15302512,
        0.29499072, 0.33932695, 0.30852377, 0.40762195, 0.40170741, 0.36259529, 0.60848355,
        0.42618036, 0.31721094, 0.02960522, 0.28256637, 0.24389413, 0.2725659,  0.10663581,
        0.27622163, 0.28264219, 0.53652936, 0.09476089, 0.40890986, 0.34848392, 0.32572666,
        0.53076893, 0.11529481, 0.29117745, 0.14625968, 0.8756339,  0.49818122, 0.10656087,
        0.1813329,  0.17664003, 0.21410346, 0.80408043, 0.02315119, 0.27155462, 0.32804728,
        0.13268511, 0.61795473, 0.49703068, 0.41696799, 0.10175809, 0.71028161, 0.29929739,
        0.17377149, 0.76075399, 0.20071237, 0.32632929, 0.36892858, 0.09416146, 0.26656723,
        0.42914796};
    EXPECT(migraphx::verify::verify_range(results_vector, s));
}

TEST_CASE(sqdiff_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("sqdiff"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {4, 4, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sqdiff_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("sqdiff"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 1};
    std::vector<float> y_data{1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {4, 4, 4};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sqrt_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data{1.02481645, 0.85643062, 0.03404123, 0.92791926, 0.10569184};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("sqrt"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sqrtf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sqrt_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    std::vector<float> input_data{1.02481645, 0.85643062, 0.03404123, 0.92791926, 0.10569184};
    mm->add_instruction(migraphx::make_op("sqrt"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {5}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return sqrtf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(squeeze_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("squeeze"), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(squeeze_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {1, 1}, {3, 3}}};
    auto p0 = mm->add_parameter("x", s1);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), p0);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(4 * 3 * 3);
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(step_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(2 * 4 * 6);
        std::iota(data.begin(), data.end(), 2);
        migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        auto r  = mm->add_instruction(
            migraphx::make_op("step", {{"axes", {0, 2, 3}}, {"steps", {2, 2, 3}}}), l0);
        mm->add_return({r});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2, 2}};
        EXPECT(result.get_shape() == s2);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(2 * 4 * 6);
        std::iota(data.begin(), data.end(), 2);
        migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        auto tl = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
        auto r = mm->add_instruction(
            migraphx::make_op("step", {{"axes", {0, 1, 2}}, {"steps", {2, 2, 3}}}), tl);
        mm->add_return({r});
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        migraphx::shape s2{migraphx::shape::float_type, {1, 2, 2, 1}};
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(sub_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("sub"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2, -2, -2};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(sub_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("sub"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 1};
    std::vector<float> y_data{1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2, -2, -2};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(tan_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("tan"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(tan_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    std::vector<float> input_data{-1, 0, 1};
    mm->add_instruction(migraphx::make_op("tan"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(tanh_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data{-1.0, 2.0, -3.0, 4.0};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("tanh"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(tanh_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    std::vector<float> input_data{-1.0, 2.0, -3.0, 4.0};
    mm->add_instruction(migraphx::make_op("tanh"), input);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return tanhf(n); });
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(topk_test)
{
    auto create_program = [](int64_t k, int64_t axis, int largest) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto data = mm->add_parameter("data", s);
        auto r    = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", axis}, {"k", k}, {"largest", largest}}), data);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({r0, r1});

        return p;
    };

    auto run_program = [&](int64_t k, int64_t axis, int largest) {
        auto p = create_program(k, axis, largest);
        p.compile(migraphx::make_target("ref"));
        std::vector<float> data = {
            2.1, 2.3, 2.0, 2.5, 1.9, 3.3, 0.2, 4.5, 0.1, 0.8, 1.0, 4.5, 2.1, 0.8, 1.5};
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        migraphx::parameter_map pp;
        pp["data"] = migraphx::argument(s, data.data());
        auto rets  = p.eval(pp);
        std::vector<float> ret_val;
        rets.front().visit([&](auto v) { ret_val.assign(v.begin(), v.end()); });
        std::vector<int64_t> ret_ind;
        rets.back().visit([&](auto v) { ret_ind.assign(v.begin(), v.end()); });

        return std::make_pair(ret_val, ret_ind);
    };

    // case 1
    {
        auto results                = run_program(4, 1, 1);
        std::vector<float> gold_val = {2.5, 2.3, 2.1, 2, 4.5, 3.3, 0.8, 0.2, 4.5, 2.1, 1.5, 1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {3, 1, 0, 2, 2, 0, 4, 1, 1, 2, 4, 0};
        EXPECT(results.second == gold_ind);
    }

    // case 2
    {
        auto results                = run_program(4, 1, 0);
        std::vector<float> gold_val = {1.9, 2, 2.1, 2.3, 0.1, 0.2, 0.8, 3.3, 0.8, 1, 1.5, 2.1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {4, 2, 0, 1, 3, 1, 4, 0, 3, 0, 4, 2};
        EXPECT(results.second == gold_ind);
    }
}

TEST_CASE(transpose_test)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);

    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
    }
    {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto l                    = mm->add_literal(migraphx::literal{a_shape, data});
        std::vector<int64_t> perm = {0, 3, 1, 2};
        auto result =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
        mm->add_instruction(migraphx::make_op("contiguous"), result);
        p.compile(migraphx::make_target("ref"));
        auto result2 = p.eval({}).back();

        std::vector<float> results_vector(12);
        result2.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
        EXPECT(migraphx::verify::verify_range(results_vector, gold));
    }
}

TEST_CASE(transpose_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}, {3, 3}}};
    auto l                    = mm->add_parameter("X", s);
    std::vector<int64_t> perm = {0, 3, 1, 2};
    mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), l);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 2, 2, 3}};
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<size_t> new_lens = {1, 3, 2, 2};
    EXPECT(result.get_shape().lens() == new_lens);

    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(unsqueeze_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<float> data(4 * 3 * 3);
        migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
        auto l0 = mm->add_literal(migraphx::literal{s1, data});
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        EXPECT(result.get_shape() == s2);
    }
}

TEST_CASE(unsqueeze_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {3, 3}, {3, 3}}};
    auto p0 = mm->add_parameter("x", s1);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), p0);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(4 * 3 * 3);
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4, 3, 3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};
    migraphx::shape sx{migraphx::shape::float_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);

    auto lb = mm->add_literal(migraphx::literal{sb, b});
    auto lx = mm->add_literal(migraphx::literal{sx, x});
    auto ly = mm->add_literal(migraphx::literal{sx, y});
    auto w  = mm->add_instruction(migraphx::make_op("where"), lb, lx, ly);
    mm->add_return({w});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify::verify_range(result_vec, gold));
}

TEST_CASE(where_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {{2, 3}, {2, 3}}};
    migraphx::shape sx{migraphx::shape::float_type, {{2, 3}, {2, 3}}};

    auto lb = mm->add_parameter("predicate", sb);
    auto lx = mm->add_parameter("X", sx);
    auto ly = mm->add_parameter("Y", sx);
    mm->add_instruction(migraphx::make_op("where"), lb, lx, ly);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> b{1, 1, 1, 0, 0, 0, 1, 0, 1};
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3, 3}};
    migraphx::shape input_fixed_shape1{migraphx::shape::uint8_type, {3, 3}};
    params["X"] = migraphx::argument(input_fixed_shape0, x.data());
    params["Y"] = migraphx::argument(input_fixed_shape0, y.data());

    params["predicate"] = migraphx::argument(input_fixed_shape1, b.data());

    auto result = p.eval(params).back();
    std::vector<float> results_vector(3 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 1, 1, 2, 2, 2, 1, 2, 1};
    EXPECT(migraphx::verify::verify_range(results_vector, gold));
}

TEST_CASE(where_broadcasted_inputs_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::bool_type, {3, 3}};

    std::vector<bool> b{true, true, true, false, false, false, true, false, true};

    auto lb  = mm->add_literal(migraphx::literal{sb, b});
    auto lx  = mm->add_literal(migraphx::literal(1.0f));
    auto ly  = mm->add_literal(migraphx::literal(2.0f));
    auto mbx = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), lx);
    auto mby = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), ly);
    auto w   = mm->add_instruction(migraphx::make_op("where"), lb, mbx, mby);
    mm->add_return({w});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vec;
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });
    std::vector<float> gold(9);
    std::vector<float> x(9, 1.0);
    std::vector<float> y(9, 2.0);
    for(int i = 0; i < gold.size(); ++i)
        gold[i] = b[i] ? x[i] : y[i];

    EXPECT(migraphx::verify::verify_range(result_vec, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
