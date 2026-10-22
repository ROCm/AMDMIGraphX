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
#include <cstdint>
#include <limits>
#include <vector>

template <class T>
static std::vector<int32_t>
eval_integer_matmul(migraphx::program& p, T a_value, T b_value, T za_value, T zb_value)
{
    const auto shapes = p.get_parameter_shapes();
    std::vector<T> a(shapes.at("a").elements(), a_value);
    std::vector<T> b(shapes.at("b").elements(), b_value);
    migraphx::parameter_map inputs{{"a", migraphx::argument(shapes.at("a"), a.data())},
                                   {"b", migraphx::argument(shapes.at("b"), b.data())}};
    if(shapes.count("za") > 0)
    {
        std::vector<T> za(shapes.at("za").elements(), za_value);
        std::vector<T> zb(shapes.at("zb").elements(), zb_value);
        inputs["za"] = migraphx::argument(shapes.at("za"), za.data());
        inputs["zb"] = migraphx::argument(shapes.at("zb"), zb.data());
        auto output  = p.eval(inputs).back();
        std::vector<int32_t> result;
        output.visit([&](auto values) { result.assign(values.begin(), values.end()); });
        return result;
    }
    auto output = p.eval(inputs).back();
    std::vector<int32_t> result;
    output.visit([&](auto values) { result.assign(values.begin(), values.end()); });
    return result;
}

static std::vector<int32_t> eval_integer_matmul_for_k(
    std::size_t k, int8_t a_value, int8_t b_value, int8_t za_value, int8_t zb_value)
{
    migraphx::onnx_options options;
    options.map_input_dims["a"] = {1, k};
    options.map_input_dims["b"] = {k, 1};
    auto p                      = read_onnx("matmulinteger_overflow_int8_150000.onnx", options);
    p.compile(migraphx::make_target("ref"));
    return eval_integer_matmul<int8_t>(p, a_value, b_value, za_value, zb_value);
}

TEST_CASE(matmulinteger_overflow_raw_dot)
{
    auto p = read_onnx("matmulinteger_overflow_int8_150000.onnx");
    p.compile(migraphx::make_target("ref"));
    EXPECT(eval_integer_matmul<int8_t>(p, 127, 127, 127, 127) == std::vector<int32_t>{0});
}

TEST_CASE(matmulinteger_overflow_correction)
{
    auto p = read_onnx("matmulinteger_overflow_int8_70000_vector.onnx");
    p.compile(migraphx::make_target("ref"));
    EXPECT(eval_integer_matmul<int8_t>(p, 127, -128, -128, -128) == std::vector<int32_t>{0});
    EXPECT(eval_integer_matmul<int8_t>(p, -1, 1, 0, 0) == std::vector<int32_t>{-70000});
}

TEST_CASE(matmulinteger_overflow_unsigned_inputs)
{
    auto p = read_onnx("matmulinteger_overflow_uint8_131072.onnx");
    p.compile(migraphx::make_target("ref"));
    EXPECT(eval_integer_matmul<uint8_t>(p, 0, 255, 0, 0) == std::vector<int32_t>{0});
}

TEST_CASE(matmulinteger_overflow_signed_path_boundary)
{
    EXPECT(eval_integer_matmul_for_k(32767, -1, 1, 0, 0) == std::vector<int32_t>{-32767});
    EXPECT(eval_integer_matmul_for_k(32768, -1, 1, 0, 0) == std::vector<int32_t>{-32768});
}

TEST_CASE(matmulinteger_overflow_dot_chunk_boundary)
{
    EXPECT(eval_integer_matmul_for_k(131068, -128, -128, 0, 0) == std::vector<int32_t>{2147418112});
    EXPECT(eval_integer_matmul_for_k(131069, -128, -128, 0, 0) == std::vector<int32_t>{2147434496});
}

TEST_CASE(matmulinteger_overflow_final_wrap)
{
    EXPECT(eval_integer_matmul_for_k(131072, -128, -128, 0, 0) ==
           std::vector<int32_t>{std::numeric_limits<int32_t>::min()});
}

TEST_CASE(matmulinteger_overflow_batched_inputs)
{
    migraphx::onnx_options options;
    options.map_input_dims["a"] = {2, 1, 150000};
    options.map_input_dims["b"] = {2, 150000, 1};
    auto p                      = read_onnx("matmulinteger_overflow_int8_150000.onnx", options);
    p.compile(migraphx::make_target("ref"));
    EXPECT(eval_integer_matmul<int8_t>(p, 127, 127, 127, 127) == std::vector<int32_t>({0, 0}));
}

TEST_CASE(matmulinteger_overflow_mismatched_k)
{
    migraphx::onnx_options options;
    options.map_input_dims["a"] = {1, 150000};
    options.map_input_dims["b"] = {160000, 1};
    EXPECT(test::throws([&] { read_onnx("matmulinteger_overflow_int8_150000.onnx", options); }));
}
