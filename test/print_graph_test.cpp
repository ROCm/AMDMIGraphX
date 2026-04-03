/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

static migraphx::program create_program()
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(basic_graph_test)
{
    migraphx::program p = create_program();

    std::stringstream ss;
    p.print_graph(ss);
    std::string test = ss.str();
    std::cout << "test = " << test << std::endl;

    EXPECT(migraphx::contains(test, "digraph"));
    EXPECT(migraphx::contains(test, "peripheries=0"));
    EXPECT(migraphx::contains(
        test,
        R"("@0"[label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" CELLSPACING="0" COLOR="transparent"><TR ALIGN="center"><TD><B>@literal</B></TD></TR></TABLE>> style="filled" fillcolor="#ADD8E6" fontcolor="#000000" color="" shape=rectangle fontname=Helvetica];)"));
    EXPECT(migraphx::contains(
        test,
        R"("y"[label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" CELLSPACING="0" COLOR="transparent"><TR ALIGN="center"><TD><B>@param</B></TD></TR><TR ALIGN="center"><TD>int64_type<BR/>{1}, {0}</TD></TR></TABLE>> style="filled" fillcolor="#F0E68C" fontcolor="#000000" color="" shape=rectangle fontname=Helvectica];)"));
    EXPECT(migraphx::contains(
        test,
        R"("x"[label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" CELLSPACING="0" COLOR="transparent"><TR ALIGN="center"><TD><B>@param</B></TD></TR><TR ALIGN="center"><TD>int64_type<BR/>{1}, {0}</TD></TR></TABLE>> style="filled" fillcolor="#F0E68C" fontcolor="#000000" color="" shape=rectangle fontname=Helvectica];)"));
    EXPECT(migraphx::contains(
        test,
        R"("@3"[label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4" CELLSPACING="0" COLOR="transparent"><TR ALIGN="center"><TD><B>sum</B></TD></TR></TABLE>> style="rounded,filled" fillcolor="#D3D3D3" fontcolor="#000000" color="" shape=none fontname=Helvetica];)"));
    EXPECT(migraphx::contains(test, R"("x" -> "@3"[label="int64_type\n{1}, {0}"];)"));
    EXPECT(migraphx::contains(test, R"("y" -> "@3"[label="int64_type\n{1}, {0}"];)"));
    EXPECT(migraphx::contains(
        test,
        R"("@4"[label=<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4" CELLSPACING="0" COLOR="transparent"><TR ALIGN="center"><TD><B>sum</B></TD></TR></TABLE>> style="rounded,filled" fillcolor="#D3D3D3" fontcolor="#000000" color="" shape=none fontname=Helvetica];)"));
    EXPECT(migraphx::contains(test, R"("@3" -> "@4"[label="int64_type\n{1}, {0}"];)"));
    EXPECT(migraphx::contains(test, R"("@0" -> "@4"[label="int64_type\n{1}, {0}"];)"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
