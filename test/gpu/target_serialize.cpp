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
#include <migraphx/target.hpp>
#include <migraphx/value.hpp>
#include "test.hpp"

TEST_CASE(gpu_target_to_value_with_options)
{
    auto t = migraphx::make_target("gpu", migraphx::value{{"gpu_arch", "gfx1100"}});
    auto v = t.to_value();
    CHECK(v.contains("gpu_arch"));
    CHECK(v.at("gpu_arch").without_key().to<std::string>() == "gfx1100");
}

TEST_CASE(gpu_target_to_value_round_trip)
{
    auto t1 = migraphx::make_target("gpu", migraphx::value{{"gpu_arch", "gfx1100"}});
    auto t2 = migraphx::make_target("gpu");
    t2.from_value(t1.to_value());
    CHECK(t2.name() == t1.name());
    CHECK(t2.to_value() == t1.to_value());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
