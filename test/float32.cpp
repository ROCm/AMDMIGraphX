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
#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"

#include <limits>

using fp32 = migraphx::generic_float<23, 8>;

#define CHECK_FLOAT(x, y) \
    CHECK(migraphx::float_equal(x, y)); \
    CHECK(migraphx::float_equal(x, y.to_float())); \
    CHECK(migraphx::float_equal(fp32{x}, y)); \
    CHECK(migraphx::float_equal(fp32{x}.to_float(), y.to_float()))


TEST_CASE(fp32_values)
{
    CHECK_FLOAT(1.0f, fp32{1.0f});
    CHECK_FLOAT(-1.0f, fp32{-1.0f});
    CHECK_FLOAT(std::numeric_limits<float>::min(), fp32::min());
    CHECK_FLOAT(std::numeric_limits<float>::lowest(), fp32::lowest());
    CHECK_FLOAT(std::numeric_limits<float>::max(), fp32::max());
    CHECK_FLOAT(std::numeric_limits<float>::epsilon(), fp32::epsilon());
    CHECK_FLOAT(std::numeric_limits<float>::infinity(), fp32::infinity());
    CHECK_FLOAT(std::numeric_limits<float>::quiet_NaN(), fp32::qnan());
    CHECK_FLOAT(std::numeric_limits<float>::signaling_NaN(), fp32::snan());
    CHECK_FLOAT(std::numeric_limits<float>::denorm_min(), fp32::denorm_min());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
