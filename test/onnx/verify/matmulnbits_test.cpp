/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "migraphx/argument.hpp"
#include "migraphx/module.hpp"
#include <cstdint>
#include <migraphx/float8.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(matmulnbits_test)
{
    auto p = optimize_onnx("matmulnbits_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> a(a_shape.elements());
    std::iota(a.begin(), a.end(), 0);
    pm["a"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {4, 1, 8}};
    std::vector<uint8_t> b{0x2,  0xe3, 0xc7, 0x89, 0xbd, 0xbe, 0x50, 0x41, 0xe9, 0xb4, 0xd4,
                           0x54, 0xc6, 0xb2, 0xfa, 0x27, 0x14, 0x3d, 0xbb, 0xe7, 0xa5, 0x0,
                           0x52, 0x28, 0xc1, 0xd9, 0x1f, 0x33, 0x16, 0x1e, 0x8b, 0x3c};
    pm["b"] = migraphx::argument(b_shape, b.data());

    auto scales_shape = migraphx::shape{migraphx::shape::float_type, {4}};
    std::vector<float> scales{1, 2, 3, 4};
    pm["scales"] = migraphx::argument(scales_shape, scales.data());

    auto zp_shape = migraphx::shape{migraphx::shape::uint8_type, {4}};
    std::vector<uint8_t> zp{0x08, 0x09, 0x0a, 0x0b};
    pm["zp"] = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        -111.0f, -290.0f, -1692.0f, -1960.0f, -335.0f, -770.0f, -4764.0f, -5992.0f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{2, 4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(matmulnbits2_test)
{
    auto p = optimize_onnx("matmulnbits2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {2, 33}};
    std::vector<float> a{
        0.15541f, 0.24434f, 0.66716f, 0.13632f, 0.76915f, 0.21328f, 0.17331f, 0.93251f, 0.14816f,
        0.08181f, 0.54035f, 0.86664f, 0.92605f, 0.89766f, 0.02441f, 0.33504f, 0.60488f, 0.25918f,
        0.64644f, 0.98881f, 0.27669f, 0.94888f, 0.21201f, 0.33377f, 0.95608f, 0.40923f, 0.66899f,
        0.58904f, 0.41560f, 0.87399f, 0.74596f, 0.10849f, 0.94527f, 0.88573f, 0.66875f, 0.57536f,
        0.81454f, 0.15699f, 0.15464f, 0.17399f, 0.08090f, 0.99368f, 0.45535f, 0.92528f, 0.91968f,
        0.76970f, 0.59638f, 0.23635f, 0.54877f, 0.96025f, 0.48969f, 0.55297f, 0.52498f, 0.29102f,
        0.01359f, 0.77372f, 0.81897f, 0.03003f, 0.00822f, 0.55477f, 0.54635f, 0.91918f, 0.76486f,
        0.73698f, 0.29821f, 0.41801};
    pm["a"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {2, 3, 8}};
    std::vector<uint8_t> b{0x18, 0x9,  0x8b, 0xe1, 0xfb, 0x94, 0x11, 0x56, 0x4e, 0xac, 0xd3, 0x4b,
                           0xf7, 0x8e, 0x54, 0xef, 0x0b, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
                           0x6e, 0xb7, 0x20, 0x4f, 0xa7, 0x82, 0x83, 0xbf, 0x20, 0xde, 0xa4, 0xf,
                           0x72, 0x81, 0x8,  0x83, 0x0a, 0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0};
    pm["b"] = migraphx::argument(b_shape, b.data());

    auto scales_shape = migraphx::shape{migraphx::shape::float_type, {6}};
    std::vector<float> scales{0.29033, 0.80435, 2.60200, 2.39623, 1.40796, 2.38139};
    pm["scales"] = migraphx::argument(scales_shape, scales.data());

    // auto zp_shape = migraphx::shape{migraphx::shape::uint8_type, {4}};
    // std::vector<uint8_t> zp{0x08, 0x09, 0x0a, 0x0b};
    // pm["zp"] = migraphx::argument{zp_shape, zp.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{18.54672f, -62.38305f, 4.978874f, -31.228657f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{2, 2});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}