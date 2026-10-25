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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

// Gold data generated with the onnx LinearAttention reference implementation
// (onnx/reference/ops/op_linear_attention.py) on the inputs below.

TEST_CASE(linear_attention_linear_test)
{
    auto p = read_onnx("linear_attention_linear_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> query_data = {
        -0.421875, 0.125, 0.234375, 0.109375,  -0.609375, -0.6875,   0.578125, -0.21875,
        0.0625,    0.5,   0.34375,  -0.890625, -0.015625, -0.046875, -0.65625, -0.109375,
        -0.984375, 1,     -0.0625,  -0.875,    0.796875,  0.890625,  -0.53125, 0.109375};
    pp["query"] = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> key_data = {-0.234375, -0.109375, -0.953125, -0.625,    -0.328125, 0.1875,
                                   -0.59375,  -0.1875,   0.25,      0.703125,  0.796875,  -0.359375,
                                   -0.765625, 0,         -0.609375, 0.734375,  -0.59375,  0.625,
                                   -0.65625,  -0.859375, 0.96875,   -0.640625, 0.953125,  0.5625};
    pp["key"]                   = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> value_data = {
        -0.109375, 0.296875, -0.5,     -0.84375,  -0.640625, 0.359375,  -0.609375, 0.4375,
        0.15625,   -0.28125, 0.015625, 0.765625,  0.203125,  -0.953125, 1,         -0.90625,
        -0.1875,   -0.96875, 0.1875,   -0.890625, 0.203125,  0.171875,  0.1875,    -0.328125};
    pp["value"] = migraphx::literal{value_shape, value_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold = {-0.006589744, 0.017886449,  0.10314851,   0.17406312,   -0.049044095,
                               0.063820444,  0.2767153,    0.15368156,   0.0127532985, -0.06896708,
                               -0.05841216,  0.17346698,   0.0017182427, -0.008062524, 0.22597724,
                               -0.20479187,  -0.052105643, -1.3364395,   -0.32230672,  -0.11201756,
                               -0.0697709,   0.37847194,   0.26049316,   -0.31607842};
    std::vector<float> gold_state = {0.27490234,
                                     -0.2578125,
                                     0.0017089844,
                                     -0.1628418,
                                     0.8508301,
                                     1.154541,
                                     0.42114258,
                                     0.17016602,
                                     0.15258789,
                                     1.4714355,
                                     -0.24731445,
                                     -0.7155762,
                                     -0.55371094,
                                     0.8239746,
                                     0.67871094,
                                     -0.0847168};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}

TEST_CASE(linear_attention_gated_test)
{
    auto p = read_onnx("linear_attention_gated_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> query_data = {-0.765625,
                                     0.125,
                                     -0.65625,
                                     -0.328125,
                                     0.171875,
                                     0.625,
                                     -0.890625,
                                     0.484375,
                                     0.171875,
                                     -0.265625,
                                     -0.015625,
                                     -0.515625};
    pp["query"]                   = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> key_data = {0.9375,
                                   0.8125,
                                   -0.421875,
                                   0.59375,
                                   -0.9375,
                                   0.671875,
                                   -0.203125,
                                   0.609375,
                                   -0.40625,
                                   -0.484375,
                                   -0.09375,
                                   0.9375};
    pp["key"]                   = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::float_type, {1, 3, 6}};
    std::vector<float> value_data = {0.046875,
                                     0.078125,
                                     0.375,
                                     -0.28125,
                                     0.484375,
                                     0.890625,
                                     -0.515625,
                                     0.25,
                                     -0.1875,
                                     -0.21875,
                                     -0.015625,
                                     -0.234375,
                                     -0.390625,
                                     -0.765625,
                                     0.828125,
                                     0.140625,
                                     0.640625,
                                     0.71875};
    pp["value"]                   = migraphx::literal{value_shape, value_data}.get_argument();
    migraphx::shape decay_shape{migraphx::shape::float_type, {1, 3, 2}};
    std::vector<float> decay_data = {-0.484375, -0.3359375, -0.25, -0.203125, -0.0625, -0.078125};
    pp["decay"]                   = migraphx::literal{decay_shape, decay_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold       = {-0.0204247,
                                     -0.034041166,
                                     -0.1633976,
                                     -0.016313864,
                                     0.0280961,
                                     0.05166057,
                                     -0.07708694,
                                     0.07452796,
                                     0.10383348,
                                     -0.18130808,
                                     0.18017021,
                                     0.26205328,
                                     0.09873889,
                                     -0.090459906,
                                     0.0661416,
                                     0.04154261,
                                     -0.29258025,
                                     -0.33966413};
    std::vector<float> gold_state = {0.64495325,
                                     0.14444534,
                                     0.085914016,
                                     -0.10837284,
                                     0.57508206,
                                     -0.29655296,
                                     0.11747426,
                                     -0.21137151,
                                     -0.30697083,
                                     -0.117499456,
                                     0.8088701,
                                     0.9409048};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}

TEST_CASE(linear_attention_gated_channel_test)
{
    auto p = read_onnx("linear_attention_gated_channel_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> query_data = {
        0.65625, -1, -0.671875, 0.671875, 0.859375, 0.453125, -0.59375, 0.265625};
    pp["query"] = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> key_data = {
        0.03125, -0.96875, -0.0625, 0.40625, 0.078125, 0.625, 0.71875, 0.515625};
    pp["key"] = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> value_data = {
        0.65625, -0.03125, -0.890625, -0.484375, 0.921875, 0.234375, -0.421875, -0.6875};
    pp["value"] = migraphx::literal{value_shape, value_data}.get_argument();
    migraphx::shape past_state_shape{migraphx::shape::float_type, {1, 2, 2, 2}};
    std::vector<float> past_state_data = {
        0.46875, -0.234375, 0.984375, -0.5625, 0.859375, -0.15625, -1, -0.71875};
    pp["past_state"] = migraphx::literal{past_state_shape, past_state_data}.get_argument();
    migraphx::shape decay_shape{migraphx::shape::float_type, {1, 2, 4}};
    std::vector<float> decay_data = {
        -0.4921875, -0.2890625, -0.4375, -0.4375, -0.2890625, -0.359375, -0.2578125, -0.0078125};
    pp["decay"] = migraphx::literal{decay_shape, decay_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold       = {0.070696734,
                                     0.20955633,
                                     -0.7686833,
                                     -0.2804099,
                                     0.39082864,
                                     -0.09505231,
                                     -0.29937676,
                                     0.04063152};
    std::vector<float> gold_state = {0.3019905,
                                     -0.08972556,
                                     0.64704466,
                                     -0.12649128,
                                     0.1685493,
                                     -0.54870313,
                                     -1.217154,
                                     -1.0101867};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}

TEST_CASE(linear_attention_delta_test)
{
    auto p = read_onnx("linear_attention_delta_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> query_data = {0.90625,
                                     0.625,
                                     0.359375,
                                     -0.21875,
                                     -0.09375,
                                     0.609375,
                                     0.703125,
                                     0.46875,
                                     -0.140625,
                                     -0.109375,
                                     0.625,
                                     -0.3125};
    pp["query"]                   = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> key_data = {-0.40625,
                                   0.96875,
                                   -0.1875,
                                   -1,
                                   -0.671875,
                                   -0.125,
                                   0.828125,
                                   -0.53125,
                                   -0.234375,
                                   0.03125,
                                   0.359375,
                                   -0.5};
    pp["key"]                   = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> value_data = {0.96875,
                                     -0.953125,
                                     0.78125,
                                     0.796875,
                                     0.53125,
                                     0.484375,
                                     -0.21875,
                                     0.140625,
                                     0.78125,
                                     -0.65625,
                                     0.171875,
                                     -0.640625};
    pp["value"]                   = migraphx::literal{value_shape, value_data}.get_argument();
    migraphx::shape beta_shape{migraphx::shape::float_type, {1, 3, 2}};
    std::vector<float> beta_data = {0.875, 0.859375, 0.828125, 0.5625, 0.921875, 0.796875};
    pp["beta"]                   = migraphx::literal{beta_shape, beta_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold       = {0.1422365,
                                     -0.13994236,
                                     0.07186037,
                                     0.07329758,
                                     0.37355587,
                                     -0.37530518,
                                     -0.34757784,
                                     -0.30631426,
                                     0.008236479,
                                     0.05537375,
                                     -0.028800469,
                                     -0.1303096};
    std::vector<float> gold_state = {-0.70308304,
                                     0.13480577,
                                     0.79746664,
                                     -0.8893017,
                                     -0.33657685,
                                     -0.44053477,
                                     -0.54281765,
                                     -0.29135564};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}

TEST_CASE(linear_attention_gated_delta_test)
{
    auto p = read_onnx("linear_attention_gated_delta_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::float_type, {1, 3, 8}};
    std::vector<float> query_data = {0.546875,  0.84375,  0.140625,  -0.875,  0.75,      -0.890625,
                                     0.765625,  0.25,     -0.578125, -0.3125, 0.015625,  -0.53125,
                                     0.34375,   0.953125, 0.890625,  0.65625, -0.359375, 0.609375,
                                     -0.921875, -0.09375, -1,        0.71875, 0.640625,  -0.578125};
    pp["query"]                   = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> key_data = {-0.515625,
                                   -0.96875,
                                   0.0625,
                                   -0.40625,
                                   0.640625,
                                   -0.71875,
                                   0.765625,
                                   0.5625,
                                   0.6875,
                                   0.703125,
                                   -0.828125,
                                   0.046875};
    pp["key"]                   = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> value_data = {0.9375,
                                     0.421875,
                                     0.21875,
                                     -0.484375,
                                     0.328125,
                                     -0.75,
                                     0.46875,
                                     -0.78125,
                                     -0.515625,
                                     -0.859375,
                                     -0.40625,
                                     -0.265625};
    pp["value"]                   = migraphx::literal{value_shape, value_data}.get_argument();
    migraphx::shape past_state_shape{migraphx::shape::float_type, {1, 2, 2, 2}};
    std::vector<float> past_state_data = {
        0.84375, -0.515625, -0.5, 0.1875, 0.46875, 0.28125, 1, -0.75};
    pp["past_state"] = migraphx::literal{past_state_shape, past_state_data}.get_argument();
    migraphx::shape decay_shape{migraphx::shape::float_type, {1, 3, 2}};
    std::vector<float> decay_data = {-0.5, -0.0625, -0.34375, -0.2890625, -0.0390625, -0.171875};
    pp["decay"]                   = migraphx::literal{decay_shape, decay_data}.get_argument();
    migraphx::shape beta_shape{migraphx::shape::float_type, {1, 3, 1}};
    std::vector<float> beta_data = {0.8125, 0.9375, 0.453125};
    pp["beta"]                   = migraphx::literal{beta_shape, beta_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold = {-0.39335006,  -0.20313817,  0.45446423,   0.045018137, -0.15808795,
                               0.28214252,   0.27349812,   0.0302478,    0.10289108,  0.15859754,
                               0.14643528,   -0.084418945, 0.284575,     -0.4005709,  0.28354633,
                               -0.39044547,  -0.15644267,  0.18526003,   0.06670217,  0.394713,
                               -0.008046061, -0.16257729,  -0.019344933, 0.140134};
    std::vector<float> gold_state = {-0.08726089,
                                     -0.86621034,
                                     -0.5649144,
                                     0.09719089,
                                     0.31537163,
                                     -0.11440441,
                                     0.41638887,
                                     -0.61156034};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}

TEST_CASE(linear_attention_decode_test)
{
    auto p = read_onnx("linear_attention_decode_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    migraphx::shape query_shape{migraphx::shape::half_type, {1, 1, 4}};
    std::vector<float> query_data = {0.65625, 0.703125, 0.234375, 0.65625};
    pp["query"]                   = migraphx::literal{query_shape, query_data}.get_argument();
    migraphx::shape key_shape{migraphx::shape::half_type, {1, 1, 2}};
    std::vector<float> key_data = {0.25, -0.03125};
    pp["key"]                   = migraphx::literal{key_shape, key_data}.get_argument();
    migraphx::shape value_shape{migraphx::shape::half_type, {1, 1, 2}};
    std::vector<float> value_data = {-0.609375, -0.984375};
    pp["value"]                   = migraphx::literal{value_shape, value_data}.get_argument();
    migraphx::shape past_state_shape{migraphx::shape::half_type, {1, 1, 2, 2}};
    std::vector<float> past_state_data = {0.171875, 0.203125, -0.59375, 0.71875};
    pp["past_state"] = migraphx::literal{past_state_shape, past_state_data}.get_argument();

    auto outputs = p.eval(pp);
    std::vector<float> result;
    outputs.front().visit([&](auto v) { result.assign(v.begin(), v.end()); });
    std::vector<float> present;
    outputs.back().visit([&](auto v) { present.assign(v.begin(), v.end()); });

    std::vector<float> gold       = {-0.27661133, 0.3527832, -0.26342773, 0.34057617};
    std::vector<float> gold_state = {0.01953125, -0.04296875, -0.57470703, 0.7495117};
    EXPECT(migraphx::verify::verify_range_with_tolerance(result, migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(present,
                                                         migraphx::verify::expected{gold_state}));
}
