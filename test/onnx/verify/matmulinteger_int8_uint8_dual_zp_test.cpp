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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <limits>
#include <onnx_test.hpp>

TEST_CASE(matmulinteger_int8_uint8_dual_zp_test)
{
    migraphx::program p = read_onnx("matmulinteger_int8_uint8_dual_zp_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::int8_type, {4, 3}};
    std::vector<int8_t> data0 = {-1, 5, -9, -2, 6, 10, -3, 7, -11, -4, 8, 0};
    migraphx::shape s1{migraphx::shape::uint8_type, {3, 2}};
    std::vector<uint8_t> data1 = {128, 129, 126, 131, 124, 133};

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(s0, data0.data());
    pp["2"] = migraphx::argument(s1, data1.data());

    auto result = p.eval(pp).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold = {-984, -1056, 1351, 1454, -1234, -1316, 117, 138};
    EXPECT(result_vector == gold);
}

template <class A, class B>
static void check_integer_correction(const std::string& name)
{
    auto p = read_onnx(name);
    p.compile(migraphx::make_target("ref"));
    auto shapes = p.get_parameter_shapes();
    std::vector<A> a(24);
    std::vector<B> b(8);
    std::vector<A> za = {std::numeric_limits<A>::min(), A{1}, std::numeric_limits<A>::max()};
    std::vector<B> zb = {std::numeric_limits<B>::max(), std::numeric_limits<B>::min()};
    if(shapes.at("za").elements() == 6)
    {
        za.push_back(A{1});
        za.push_back(std::numeric_limits<A>::max());
        za.push_back(std::numeric_limits<A>::min());
    }
    for(std::size_t i = 0; i < a.size(); ++i)
        a[i] = i % 2 == 0 ? std::numeric_limits<A>::min() : std::numeric_limits<A>::max();
    for(std::size_t i = 0; i < b.size(); ++i)
        b[i] = i % 3 == 0 ? std::numeric_limits<B>::min() : std::numeric_limits<B>::max();
    migraphx::parameter_map inputs{{"a", migraphx::argument(shapes.at("a"), a.data())},
                                   {"b", migraphx::argument(shapes.at("b"), b.data())},
                                   {"za", migraphx::argument(shapes.at("za"), za.data())},
                                   {"zb", migraphx::argument(shapes.at("zb"), zb.data())}};
    std::vector<int32_t> expected;
    for(std::size_t batch = 0; batch < 2; ++batch)
        for(std::size_t row = 0; row < 3; ++row)
            for(std::size_t col = 0; col < 2; ++col)
            {
                int32_t sum = 0;
                for(std::size_t k = 0; k < 4; ++k)
                    sum += (int32_t{a[batch * 12 + row * 4 + k]} -
                            int32_t{za[(za.size() == 6 ? batch * 3 : 0) + row]}) *
                           (int32_t{b[k * 2 + col]} - int32_t{zb[col]});
                expected.push_back(sum);
            }
    auto output = p.eval(inputs).back();
    std::vector<int32_t> actual;
    output.visit([&](auto values) { actual.assign(values.begin(), values.end()); });
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>({2, 3, 2}));
    EXPECT(actual == expected);
}

TEST_CASE(matmulinteger_int32_correction_batched_extrema)
{
    check_integer_correction<int8_t, int8_t>("matmulinteger_correction_int8_int8.onnx");
    check_integer_correction<int8_t, uint8_t>("matmulinteger_correction_int8_uint8.onnx");
    check_integer_correction<uint8_t, int8_t>("matmulinteger_correction_uint8_int8.onnx");
    check_integer_correction<uint8_t, uint8_t>("matmulinteger_correction_uint8_uint8.onnx");
    check_integer_correction<int8_t, int8_t>("matmulinteger_correction_batched_int8_int8.onnx");
    check_integer_correction<int8_t, uint8_t>("matmulinteger_correction_batched_int8_uint8.onnx");
    check_integer_correction<uint8_t, int8_t>("matmulinteger_correction_batched_uint8_int8.onnx");
    check_integer_correction<uint8_t, uint8_t>("matmulinteger_correction_batched_uint8_uint8.onnx");
}

template <class A, class B>
static void check_integer_vector(const std::string& name, bool lhs)
{
    auto p = read_onnx(name);
    p.compile(migraphx::make_target("ref"));
    auto shapes = p.get_parameter_shapes();
    std::vector<A> a(shapes.at("a").elements());
    std::vector<B> b(shapes.at("b").elements());
    A za = std::numeric_limits<A>::max();
    B zb = std::numeric_limits<B>::min();
    for(std::size_t i = 0; i < a.size(); ++i)
        a[i] = i % 2 == 0 ? std::numeric_limits<A>::min() : std::numeric_limits<A>::max();
    for(std::size_t i = 0; i < b.size(); ++i)
        b[i] = i % 3 == 0 ? std::numeric_limits<B>::min() : std::numeric_limits<B>::max();
    migraphx::parameter_map inputs{{"a", migraphx::argument(shapes.at("a"), a.data())},
                                   {"b", migraphx::argument(shapes.at("b"), b.data())},
                                   {"za", migraphx::argument(shapes.at("za"), &za)},
                                   {"zb", migraphx::argument(shapes.at("zb"), &zb)}};
    std::vector<int32_t> expected(2, 0);
    for(std::size_t i = 0; i < 2; ++i)
        for(std::size_t k = 0; k < 4; ++k)
            expected[i] += (int32_t{a[lhs ? k : i * 4 + k]} - int32_t{za}) *
                           (int32_t{b[lhs ? k * 2 + i : k]} - int32_t{zb});
    auto output = p.eval(inputs).back();
    std::vector<int32_t> actual;
    output.visit([&](auto values) { actual.assign(values.begin(), values.end()); });
    EXPECT(output.get_shape().lens() == std::vector<std::size_t>({2}));
    EXPECT(actual == expected);
}

TEST_CASE(matmulinteger_int32_correction_vectors)
{
    for(bool lhs : {false, true})
    {
        const std::string prefix = lhs ? "matmulinteger_vector_lhs_" : "matmulinteger_vector_rhs_";
        check_integer_vector<int8_t, int8_t>(prefix + "int8_int8.onnx", lhs);
        check_integer_vector<int8_t, uint8_t>(prefix + "int8_uint8.onnx", lhs);
        check_integer_vector<uint8_t, int8_t>(prefix + "uint8_int8.onnx", lhs);
        check_integer_vector<uint8_t, uint8_t>(prefix + "uint8_uint8.onnx", lhs);
    }
}

TEST_CASE(matmulinteger_int32_correction_boundary_test)
{
    auto p = read_onnx("matmulinteger_int32_correction_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::uint8_type, {1, 4}};
    for(auto& test : std::vector<std::pair<std::vector<uint8_t>, std::vector<int32_t>>>{
            {{255, 255, 255, 255}, {508, 1270}},
            {{0, 0, 0, 0}, {-2, -5}},
            {{24, 115, 107, 231}, {-359, 348}}})
    {
        migraphx::parameter_map inputs;
        inputs["a"] = migraphx::argument(s, test.first.data());
        auto output = p.eval(inputs).back();
        std::vector<int32_t> actual;
        output.visit([&](auto values) { actual.assign(values.begin(), values.end()); });
        EXPECT(actual == test.second);
    }
}
