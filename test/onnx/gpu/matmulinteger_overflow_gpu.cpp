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
#include <onnx_test.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

template <class A, class B>
static std::vector<int32_t> eval_matmulinteger_gpu(
    migraphx::program& p, A a_value, B b_value, A za_value = A{}, B zb_value = B{})
{
    auto shapes  = p.get_parameter_shapes();
    auto a       = std::vector<A>(shapes.at("a").elements(), a_value);
    auto b       = std::vector<B>(shapes.at("b").elements(), b_value);
    auto a_shape = shapes.at("a");
    auto b_shape = shapes.at("b");
    migraphx::parameter_map params{{"a", migraphx::argument(a_shape, a.data())},
                                   {"b", migraphx::argument(b_shape, b.data())}};
    std::vector<A> za;
    std::vector<B> zb;
    if(shapes.count("za") > 0)
    {
        za           = std::vector<A>(shapes.at("za").elements(), za_value);
        zb           = std::vector<B>(shapes.at("zb").elements(), zb_value);
        params["za"] = migraphx::argument(shapes.at("za"), za.data());
        params["zb"] = migraphx::argument(shapes.at("zb"), zb.data());
    }
    migraphx::argument result = p.eval(params).back();
    std::vector<int32_t> output;
    result.visit([&](auto values) { output.assign(values.begin(), values.end()); });
    return output;
}

TEST_CASE(matmulinteger_overflow_gpu_raw_dot)
{
    auto p = read_onnx("matmulinteger_overflow_int8_150000.onnx");
    p.compile(migraphx::make_target("gpu"), {true});
    auto result = eval_matmulinteger_gpu<int8_t, int8_t>(p, 127, 127, 127, 127);
    EXPECT(result == std::vector<int32_t>{0});
}

TEST_CASE(matmulinteger_overflow_gpu_uint8)
{
    auto p = read_onnx("matmulinteger_overflow_uint8_131072.onnx");
    p.compile(migraphx::make_target("gpu"), {true});
    auto result = eval_matmulinteger_gpu<uint8_t, uint8_t>(p, 255, 255);
    // (255 * 255 * 131072) wraps to -66977792 in the int32 output.
    EXPECT(result == std::vector<int32_t>{-66977792});
}

TEST_CASE(matmulinteger_overflow_gpu_asymmetric_zero_point)
{
    constexpr std::size_t k = 131069;
    migraphx::onnx_options options;
    options.map_input_dims["a"] = {k};
    options.map_input_dims["b"] = {k};
    auto p = read_onnx("matmulinteger_overflow_int8_70000_vector.onnx", options);
    p.compile(migraphx::make_target("gpu"), {true});
    auto result = eval_matmulinteger_gpu<int8_t, int8_t>(p, 1, 1, 0, -128);
    // Each output is (1 - 0) * (1 - -128) * K.
    EXPECT(result == std::vector<int32_t>{16907901});
}

TEST_CASE(matmulinteger_overflow_gpu_mixed_batched_zero_points)
{
    constexpr std::size_t k = 131069;
    migraphx::onnx_options options;
    options.map_input_dims["a"]  = {2, 1, 2, k};
    options.map_input_dims["b"]  = {3, k, 2};
    options.map_input_dims["za"] = {1, 1, 2, 1};
    options.map_input_dims["zb"] = {1, 1, 2};
    auto p = read_onnx("matmulinteger_correction_batched_int8_uint8.onnx", options);
    p.compile(migraphx::make_target("gpu"), {true});

    auto shapes        = p.get_parameter_shapes();
    const auto a_shape = shapes.at("a");
    std::vector<int8_t> a(a_shape.elements());
    std::generate(a.begin(), a.end(), [i = std::size_t{0}, &a_shape]() mutable {
        const auto row = a_shape.multi(i++)[2];
        return row == 0 ? int8_t{127} : int8_t{-128};
    });
    const auto b_shape = shapes.at("b");
    std::vector<uint8_t> b(b_shape.elements());
    std::generate(b.begin(), b.end(), [i = std::size_t{0}, &b_shape]() mutable {
        return b_shape.multi(i++)[2] == 0 ? uint8_t{255} : uint8_t{128};
    });
    std::vector<int8_t> za{-128, 127};
    std::vector<uint8_t> zb{128, 255};

    migraphx::parameter_map params{{"a", migraphx::argument(shapes.at("a"), a.data())},
                                   {"b", migraphx::argument(shapes.at("b"), b.data())},
                                   {"za", migraphx::argument(shapes.at("za"), za.data())},
                                   {"zb", migraphx::argument(shapes.at("zb"), zb.data())}};
    migraphx::argument result = p.eval(params).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>({2, 3, 2, 2}));
    std::vector<int32_t> output;
    result.visit([&](auto values) { output.assign(values.begin(), values.end()); });

    // 255 * 127 * K wraps to -50297731; changing the sign gives 50297731.
    const std::array<int32_t, 4> pattern{-50297731, 50297731, 50297731, -50297731};
    std::vector<int32_t> expected(24);
    std::generate(expected.begin(), expected.end(), [i = std::size_t{0}, &pattern]() mutable {
        return pattern[i++ % pattern.size()];
    });
    EXPECT(output == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
