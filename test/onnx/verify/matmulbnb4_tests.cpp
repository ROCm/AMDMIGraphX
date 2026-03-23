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
    std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    pm["A"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {16}};
    std::vector<uint8_t> b{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                           0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 2.0f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    std::cout << "matmulbnb4_fp4_test results: ";
    for(const auto& val : result_vector) {
        std::cout << val << "f, ";
    }
    std::cout << std::endl;
    
    std::vector<float> gold{-4.0f, -8.0f, -8.0f, 0.0f,
                            -36.0f, -72.0f, -72.0f, 0.0f};

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
    std::vector<uint8_t> b{
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0x02, 0x13, 0x24, 0x35, 0x46, 0x57, 0x68, 0x79,
        0x8a, 0x9b, 0xac, 0xbd, 0xce, 0xdf, 0xe0, 0xf1,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00,
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
        0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f
    };
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {8}};
    std::vector<float> absmax{1.0f, 1.5f, 2.0f, 2.5f, 1.2f, 1.8f, 2.2f, 2.8f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    std::cout << "matmulbnb4_nf4_test results: ";
    for(const auto& val : result_vector) {
        std::cout << val << "f, ";
    }
    std::cout << std::endl;
    
    std::vector<float> gold{
        -5.89f, -7.32f, -9.36f, -11.15f, -5.65f, -8.13f, -10.28f, -12.69f,
        -21.89f, -31.82f, -41.36f, -50.65f, -24.05f, -34.53f, -44.28f, -54.69f,
        -37.89f, -56.32f, -73.36f, -90.15f, -42.45f, -60.93f, -78.28f, -96.69f
    };

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
    std::vector<uint8_t> b{
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
        0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00
    };
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 1.5f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    std::cout << "matmulbnb4_block32_test results: ";
    for(const auto& val : result_vector) {
        std::cout << val << "f, ";
    }
    std::cout << std::endl;
    
    std::vector<float> gold{
        -8.0f, -16.0f, -24.0f, -24.0f,
        -40.0f, -80.0f, -120.0f, -120.0f,
        -72.0f, -144.0f, -216.0f, -216.0f
    };

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
    std::vector<uint8_t> b{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                           0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 1.5f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    std::cout << "matmulbnb4_fp4_1d_input_test results: ";
    for(const auto& val : result_vector) {
        std::cout << val << "f, ";
    }
    std::cout << std::endl;
    
    std::vector<float> gold{-4.0f, -8.0f, -12.0f, -12.0f};

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
    std::vector<uint8_t> b{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                           0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    pm["B"] = migraphx::argument(b_shape, b.data());

    auto absmax_shape = migraphx::shape{migraphx::shape::float_type, {2}};
    std::vector<float> absmax{1.0f, 2.0f};
    pm["absmax"] = migraphx::argument(absmax_shape, absmax.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    
    std::cout << "matmulbnb4_fp4_3d_input_test results: ";
    for(const auto& val : result_vector) {
        std::cout << val << "f, ";
    }
    std::cout << std::endl;
    
    std::vector<float> gold{
        -4.0f, -8.0f, -8.0f, 0.0f,
        -36.0f, -72.0f, -72.0f, 0.0f,
        -68.0f, -136.0f, -136.0f, 0.0f,
        -100.0f, -200.0f, -200.0f, 0.0f,
        -132.0f, -264.0f, -264.0f, 0.0f,
        -164.0f, -328.0f, -328.0f, 0.0f
    };

    EXPECT(result.get_shape().lens() == std::vector<size_t>{2, 3, 4});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
