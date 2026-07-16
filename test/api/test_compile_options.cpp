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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

static migraphx::program create_add_program(const migraphx::shape& s)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    std::vector<float> x_values(s.elements(), 1);
    auto x = m.add_literal(s, x_values.data());
    std::vector<float> y_values(s.elements(), -1);
    auto y = m.add_literal(s, y_values.data());
    auto r = m.add_instruction(migraphx::operation("add"), {x, y});
    m.add_return({r});
    return p;
}

static void run_and_check(migraphx::program& p, const migraphx::shape& s)
{
    migraphx::program_parameters pp;
    auto outputs = p.eval(pp);
    std::vector<float> expected(s.elements(), 0);
    CHECK(outputs[0] == migraphx::argument(s, expected.data()));
}

TEST_CASE(compile_options_flags)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_offload_copy(false);
    options.set_fast_math(false);
    options.set_exhaustive_tune_flag(false);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_option_scalar)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_option("int_option", 42);
    options.set_advance_backend_option("str_option", "gfx942");
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_option_vector)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_option("ints", std::vector<int>{1, 2, 3});
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_options_json)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_options("{a:%i, b:%i}", 1, 2);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
