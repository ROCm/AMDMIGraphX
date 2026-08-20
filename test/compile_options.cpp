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
#include <migraphx/compile_options.hpp>
#include <migraphx/errors.hpp>
#include "test.hpp"

TEST_CASE(set_backend_options_merges_object)
{
    migraphx::value v;
    v["int_option"] = 42;
    v["str_option"] = "gfx942";
    v["ints"]       = {1, 2, 3};

    migraphx::compile_options options;
    migraphx::set_backend_options(options, v);

    EXPECT(options.backend_options.size() == 3);
    EXPECT(options.backend_options.at("int_option") == migraphx::value(42));
    EXPECT(options.backend_options.at("str_option") == migraphx::value("gfx942"));
    EXPECT(options.backend_options.at("ints") == migraphx::value({1, 2, 3}));
}

TEST_CASE(set_backend_options_distinguishes_problem_cache_tiers)
{
    // The read-only and writable problem-cache options must parse as two
    // distinct backend-option entries (the GPU target routes each tier).
    migraphx::value v;
    v["problem_cache_files"]          = "ship.json";
    v["writable_problem_cache_files"] = "dev.json";

    migraphx::compile_options options;
    migraphx::set_backend_options(options, v);

    EXPECT(options.backend_options.size() == 2);
    EXPECT(options.backend_options.at("problem_cache_files") == migraphx::value("ship.json"));
    EXPECT(options.backend_options.at("writable_problem_cache_files") ==
           migraphx::value("dev.json"));
}

TEST_CASE(set_backend_options_overwrites_existing)
{
    migraphx::compile_options options;
    options.backend_options["x"] = 1;

    migraphx::value v;
    v["x"] = 2;
    v["y"] = 3;
    migraphx::set_backend_options(options, v);

    EXPECT(options.backend_options.size() == 2);
    EXPECT(options.backend_options.at("x") == migraphx::value(2));
    EXPECT(options.backend_options.at("y") == migraphx::value(3));
}

TEST_CASE(set_backend_options_empty)
{
    migraphx::compile_options options;
    migraphx::set_backend_options(options, migraphx::value::object{});
    EXPECT(options.backend_options.empty());
}

TEST_CASE(set_backend_options_non_object_throws)
{
    migraphx::compile_options options;

    // null
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::set_backend_options(options, migraphx::value{}); },
        "expects an object value"));

    // scalar
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::set_backend_options(options, migraphx::value(42)); },
        "expects an object value"));

    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::set_backend_options(options, migraphx::value("gfx942")); },
        "expects an object value"));

    // array
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::set_backend_options(options, migraphx::value({1, 2, 3})); },
        "expects an object value"));

    // nothing was merged on any of the failed calls
    EXPECT(options.backend_options.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
