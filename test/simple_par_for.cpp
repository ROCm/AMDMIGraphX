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
#include <migraphx/par_for.hpp>
#include <migraphx/errors.hpp>
#include <algorithm>
#include <atomic>
#include <vector>

#include <test.hpp>

TEST_CASE(par_for_runs_all)
{
    std::vector<int> data(64, 0);
    migraphx::par_for(data.size(), 1, [&](std::size_t i) { data[i] = 1; });
    EXPECT(std::all_of(data.begin(), data.end(), [](int x) { return x == 1; }));
}

TEST_CASE(par_for_exception_propagates)
{
    EXPECT(test::throws<migraphx::exception>(
        [] {
            migraphx::par_for(64, 1, [](std::size_t i) {
                if(i % 2 == 0)
                    MIGRAPHX_THROW("par_for_error");
            });
        },
        "par_for_error"));
}

TEST_CASE(par_for_exception_propagates_serial)
{
    EXPECT(test::throws<migraphx::exception>(
        [] {
            migraphx::par_for(4, 100, [](std::size_t i) {
                if(i == 2)
                    MIGRAPHX_THROW("par_for_error");
            });
        },
        "par_for_error"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
