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
#include <migraphx/gpu/target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/context.hpp>
#include <test.hpp>
#include <cstdlib>

TEST_CASE(tuning_compile_budget_rejects_negative)
{
    migraphx::gpu::target t;
    auto ctx = t.get_context();
    EXPECT(test::throws<migraphx::exception>(
        [&] { t.get_passes(ctx, migraphx::compile_options{}); },
        "MIGRAPHX_TUNING_COMPILE_BUDGET must be 0 or a positive number of milliseconds, not -1"));
}

// The variable is read once per process, so it is set before any test runs
int main(int argc, const char* argv[])
{
#ifdef _WIN32
    _putenv_s("MIGRAPHX_TUNING_COMPILE_BUDGET", "-1");
#else
    setenv("MIGRAPHX_TUNING_COMPILE_BUDGET", "-1", 1);
#endif
    test::run(argc, argv);
}
