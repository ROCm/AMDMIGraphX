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

#include <test.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <cstdlib>
#include <string>

// MIGRAPHX_GPU_HIP_FLAGS is read once, on the first call in this process.
TEST_CASE(quoted_hip_flag_stays_one_argument)
{
    constexpr const char* name = "MIGRAPHX_GPU_HIP_FLAGS";
#ifdef _WIN32
    _putenv_s(name, R"(-I"/opt/custom headers" -DNAME=1 -I/opt/third\ headers)");
#else
    setenv(name, R"(-I"/opt/custom headers" -DNAME=1 -I/opt/third\ headers)", 1);
#endif

    auto options = migraphx::gpu::compile_hip_options({}, "gfx908");

#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif

    EXPECT(options.size() >= 3);
    EXPECT(options[options.size() - 3] == "-I/opt/custom headers");
    EXPECT(options[options.size() - 2] == "-DNAME=1");
    EXPECT(options.back() == "-I/opt/third headers");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
