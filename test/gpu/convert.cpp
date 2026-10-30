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
#include <algorithm>
#include <cmath>
#include <limits>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <test.hpp>

TEST_CASE(convert_nonsaturating_half_runtime_test)
{
    migraphx::program p;
    auto* mm       = p.get_main_module();
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    // Round-to-nearest overflows at 65520, not immediately above the maximum finite half.
    std::vector<float> data{131008,
                            -131008,
                            65504,
                            -65504,
                            65519,
                            -65519,
                            65520,
                            -65520,
                            65536,
                            -65536,
                            0.0f,
                            -0.0f,
                            inf,
                            -inf,
                            nan};
    migraphx::shape shape{migraphx::shape::float_type, {data.size()}};
    auto x = mm->add_parameter("x", shape);
    auto y = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::shape::half_type}, {"saturate", false}}),
        x);
    mm->add_return({y});
    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);
    auto result = p.eval({{"x", migraphx::argument{shape, data.data()}}}).back();
    std::vector<float> values;
    result.visit([&](auto output) { values.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        inf, -inf, 65504, -65504, 65504, -65504, inf, -inf, inf, -inf, 0.0f, -0.0f, inf, -inf};
    CHECK(values.size() == 15);
    EXPECT(std::equal(gold.begin(), gold.end(), values.begin()));
    EXPECT(not std::signbit(values.at(10)));
    EXPECT(std::signbit(values.at(11)));
    EXPECT(std::isnan(values.at(14)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
