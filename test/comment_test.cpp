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

#include "test.hpp"

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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
