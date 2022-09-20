/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test.hpp"
#include <migraphx/check_shapes.hpp>
#include <migraphx/make_op.hpp>

/*!
 * Tests for check_shapes object handling dynamic shapes
 */

using migraphx::shape;

bool create_shapes(bool dynamic_allowed)
{
    try
    {
        shape a{shape::int64_type, {3}};
        shape b{shape::float_type, {{3, 6, 0}, {4, 4, 0}}};
        auto op = migraphx::make_op("add");
        migraphx::check_shapes{{a, b}, op, dynamic_allowed}.has(2);
        return true;
    }
    catch(...)
    {
        return false;
    }
}

TEST_CASE(allow_dynamic_shape) { EXPECT(create_shapes(true)); }

TEST_CASE(fail_dynamic_shape) { EXPECT(not create_shapes(false)); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
