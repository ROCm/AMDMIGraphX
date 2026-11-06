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

#include "migraphx/argument.hpp"
#include "migraphx/module.hpp"
#include <cstdint>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(matmulbnb4_fp4_test)
{
    auto p = optimize_onnx("matmulbnb4_fp4_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {2, 8}};
    std::vector<float> a{1.0f,
                         2.0f,
                         3.0f,
                         4.0f,
                         5.0f,
                         6.0f,
                         7.0f,
                         8.0f,
                         9.0f,
                         10.0f,
                         11.0f,
                         12.0f,
                         13.0f,
                         14.0f,
                         15.0f,
                         16.0f};
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {16}};
    std::vector<uint8_t> b{0x12,
                           0x34,
                           0x56,
                           0x78,
                           0x9a,
                           0xbc,
                           0xde,
                           0xf0,
                           0x11,
                           0x22,
                           0x33,
                           0x44,
                           0x55,
                           0x66,
                           0x77,
                           0x88};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 2.0f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{25.0f, 47.0f, 27.5f, 63.5f, 61.0f, 131.0f, 67.5f, 167.5f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{2, 4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(matmulbnb4_nf4_test)
{
    auto p = optimize_onnx("matmulbnb4_nf4_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {3, 16}};
    std::vector<float> a(a_shape.elements());
    std::iota(a.begin(), a.end(), 0.5f);
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> b{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x10, 0x32, 0x54,
                           0x76, 0x98, 0xba, 0xdc, 0xfe, 0x02, 0x13, 0x24, 0x35, 0x46, 0x57,
                           0x68, 0x79, 0x8a, 0x9b, 0xac, 0xbd, 0xce, 0xdf, 0xe0, 0xf1, 0x11,
                           0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
                           0xdd, 0xee, 0xff, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde,
                           0xf0, 0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {8}};
    std::vector<float> absmax{1.0f, 1.5f, 2.0f, 2.5f, 1.2f, 1.8f, 2.2f, 2.8f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{39.4385f,  60.953f,  -34.0385f, 102.864f, -18.499f,  60.0836f,
                            56.007f,   67.9306f, 45.4285f,  69.9379f, -175.357f, 309.462f,
                            -99.0187f, 202.427f, 69.1849f,  84.7025f, 51.4185f,  78.9229f,
                            -316.675f, 516.06f,  -179.538f, 344.771f, 82.3629f,  101.474f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{3, 8});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(matmulbnb4_block32_test)
{
    auto p = optimize_onnx("matmulbnb4_block32_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {3, 16}};
    std::vector<float> a(a_shape.elements());
    std::iota(a.begin(), a.end(), 1.0f);
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {32}};
    std::vector<uint8_t> b{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x21, 0x43, 0x65,
                           0x87, 0xa9, 0xcb, 0xed, 0x0f, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                           0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 1.5f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{156.0f,
                            155.0f,
                            146.25f,
                            257.25f,
                            396.0f,
                            395.0f,
                            362.25f,
                            761.25f,
                            636.0f,
                            635.0f,
                            578.25f,
                            1265.25f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{3, 4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(matmulbnb4_fp4_1d_input_test)
{
    auto p = optimize_onnx("matmulbnb4_fp4_1d_input_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {8}};
    std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {16}};
    std::vector<uint8_t> b{0x01,
                           0x23,
                           0x45,
                           0x67,
                           0x89,
                           0xab,
                           0xcd,
                           0xef,
                           0x10,
                           0x32,
                           0x54,
                           0x76,
                           0x98,
                           0xba,
                           0xdc,
                           0xfe};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 1.5f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{20.5f, 56.5f, 31.5f, 85.5f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(matmulbnb4_fp4_3d_input_test)
{
    auto p = optimize_onnx("matmulbnb4_fp4_3d_input_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {2, 3, 8}};
    std::vector<float> a(a_shape.elements());
    std::iota(a.begin(), a.end(), 1.0f);
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {16}};
    std::vector<uint8_t> b{0x12,
                           0x34,
                           0x56,
                           0x78,
                           0x9a,
                           0xbc,
                           0xde,
                           0xf0,
                           0x11,
                           0x22,
                           0x33,
                           0x44,
                           0x55,
                           0x66,
                           0x77,
                           0x88};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 2.0f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{25.0f,  47.0f,  27.5f,  63.5f,  61.0f,  131.0f, 67.5f,  167.5f,
                            97.0f,  215.0f, 107.5f, 271.5f, 133.0f, 299.0f, 147.5f, 375.5f,
                            169.0f, 383.0f, 187.5f, 479.5f, 205.0f, 467.0f, 227.5f, 583.5f};

    EXPECT(result.get_shape().lens() == std::vector<size_t>{2, 3, 4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
