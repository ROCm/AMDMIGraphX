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
#include <string>
#include "test.hpp"

static migraphx::program make_add_sub_program()
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    std::vector<float> x_values(9, 1);
    std::vector<float> y_values(9, 2);
    auto x   = m.add_literal(s, x_values.data());
    auto y   = m.add_literal(s, y_values.data());
    auto sum = m.add_instruction(migraphx::operation("add"), {x, y});
    auto sub = m.add_instruction(migraphx::operation("sub"), {sum, x});
    m.add_return({sub});
    p.compile(migraphx::target("ref"));
    return p;
}

TEST_CASE(run_trace)
{
    auto p            = make_add_sub_program();
    size_t call_count = 0;
    migraphx::program_parameters pp;
    p.run_trace(pp, [&](size_t, const char*, const migraphx::argument&) { call_count++; });
    CHECK(call_count > 0);
}

TEST_CASE(run_trace_filter_by_name)
{
    auto p = make_add_sub_program();
    std::vector<float> captured;
    migraphx::program_parameters pp;
    p.run_trace(pp, [&](size_t, const char* name, const migraphx::argument& output) {
        if(std::string(name).find("sub") != std::string::npos)
            captured = output.as_vector<float>();
    });
    std::vector<float> expected(9, 2);
    CHECK(captured == expected);
}

TEST_CASE(run_trace_filter_by_index)
{
    auto p = make_add_sub_program();
    std::vector<float> captured;
    migraphx::program_parameters pp;
    p.run_trace(pp, [&](size_t idx, const char*, const migraphx::argument& output) {
        if(idx == 3) // sub instruction, same as filter_by_name test
            captured = output.as_vector<float>();
    });
    std::vector<float> expected(9, 2);
    CHECK(captured == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
