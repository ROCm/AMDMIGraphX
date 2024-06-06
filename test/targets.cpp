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
#include <migraphx/target.hpp>
#include "test.hpp"

bool verify_target(const std::string& name)
{
    auto t = migraphx::make_target(name);
    return t.name() == name;
}

TEST_CASE(make_target)
{
    CHECK(verify_target("ref"));
#ifdef HAVE_CPU
    CHECK(verify_target("cpu"));
#endif
#ifdef HAVE_GPU
    CHECK(verify_target("gpu"));
#endif
#ifdef HAVE_FPGA
    CHECK(verify_target("fpga"));
#endif
}

TEST_CASE(make_invalid_target)
{
    EXPECT(test::throws([&] { migraphx::make_target("mi100"); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
