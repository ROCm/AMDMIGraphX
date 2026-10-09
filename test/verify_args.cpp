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

#include <migraphx/verify_args.hpp>
#include <migraphx/argument.hpp>
#include <test.hpp>

TEST_CASE(verify_args_empty)
{
    migraphx::shape s{migraphx::shape::float_type, {0, 4}};
    migraphx::argument target{s, nullptr};
    migraphx::argument ref{s, nullptr};
    EXPECT(migraphx::verify_args(
        "empty", target, migraphx::verify::expected{ref}, migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(verify_args_empty_different_strides)
{
    migraphx::shape target_shape{migraphx::shape::float_type, {0, 4}, {4, 1}};
    migraphx::shape ref_shape{migraphx::shape::float_type, {0, 4}, {8, 1}};
    EXPECT(migraphx::verify_args("empty strides",
                                 migraphx::argument{target_shape, nullptr},
                                 migraphx::verify::expected{migraphx::argument{ref_shape, nullptr}},
                                 migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(verify_args_empty_mismatch)
{
    migraphx::shape empty_shape{migraphx::shape::float_type, {0, 4}};
    migraphx::shape nonempty_shape{migraphx::shape::float_type, {1, 4}};
    EXPECT(not migraphx::verify_args(
        "empty mismatch",
        migraphx::argument{empty_shape, nullptr},
        migraphx::verify::expected{migraphx::argument{nonempty_shape, nullptr}},
        migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(verify_args_empty_shape_mismatch)
{
    migraphx::shape target_shape{migraphx::shape::float_type, {0, 4}};
    migraphx::shape ref_shape{migraphx::shape::float_type, {0, 5}};
    EXPECT(not migraphx::verify_args(
        "empty shape mismatch",
        migraphx::argument{target_shape, nullptr},
        migraphx::verify::expected{migraphx::argument{ref_shape, nullptr}},
        migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(verify_args_empty_type_mismatch)
{
    migraphx::shape target_shape{migraphx::shape::float_type, {0, 4}};
    migraphx::shape ref_shape{migraphx::shape::int32_type, {0, 4}};
    EXPECT(not migraphx::verify_args(
        "empty type mismatch",
        migraphx::argument{target_shape, nullptr},
        migraphx::verify::expected{migraphx::argument{ref_shape, nullptr}},
        migraphx::verify::tolerance{1e-3}));
}

TEST_CASE(verify_args_empty_with_tolerance)
{
    migraphx::shape s{migraphx::shape::float_type, {0, 4}};
    EXPECT(migraphx::verify_args_with_tolerance(
        "empty tolerance",
        migraphx::argument{s, nullptr},
        migraphx::verify::expected{migraphx::argument{s, nullptr}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
