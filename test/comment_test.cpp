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
#include <migraphx/builtin.hpp>
#include <migraphx/context.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/stringutils.hpp>

#include "test.hpp"

#include <cstdio>
#include <sstream>

TEST_CASE(comment_compute)
{
    migraphx::builtin::comment op{"sample comment"};
    migraphx::context ctx;
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto result = op.compute(ctx, s, {});
    EXPECT(result.get_shape() == s);
    EXPECT(result.data() == nullptr);
}

TEST_CASE(comment_compute_empty_shape)
{
    migraphx::builtin::comment op{"empty shape comment"};
    migraphx::context ctx;
    migraphx::shape s{};
    auto result = op.compute(ctx, s, {});
    EXPECT(result.get_shape() == s);
    EXPECT(result.data() == nullptr);
}

TEST_CASE(comment_stream_operator)
{
    migraphx::builtin::comment op{"streamed text"};
    std::ostringstream ss;
    ss << op;
    EXPECT(ss.str() == "@comment: streamed text");
}

static migraphx::program create_program_with_comment()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_instruction(migraphx::make_op("@comment", {{"text", "gpu::mlir_op test"}}), {});
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto two = mm->add_literal(2);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, two);
    mm->add_return({add});
    return p;
}

TEST_CASE(comment_eval)
{
    migraphx::program p = create_program_with_comment();
    auto result         = p.eval({{"x", migraphx::literal{3}.get_argument()}}).back();
    EXPECT(result == migraphx::literal{5});
}

TEST_CASE(comment_as_value)
{
    migraphx::program p1 = create_program_with_comment();
    migraphx::program p2;
    p2.from_value(p1.to_value());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(comment_as_msgpack)
{
    migraphx::file_options options;
    options.format           = "msgpack";
    migraphx::program p1     = create_program_with_comment();
    std::vector<char> buffer = migraphx::save_buffer(p1, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(comment_as_file)
{
    std::string filename = "migraphx_comment_program.mxr";
    migraphx::program p1 = create_program_with_comment();
    migraphx::save(p1, filename);
    migraphx::program p2 = migraphx::load(filename);
    std::remove(filename.c_str());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(comment_print)
{
    migraphx::program p = create_program_with_comment();
    auto s              = migraphx::to_string(p);
    EXPECT(s.find("@comment") != std::string::npos);
    EXPECT(s.find("gpu::mlir_op test") != std::string::npos);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
