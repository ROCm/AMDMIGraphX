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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

struct test_gather_flatten_stride_slice : verify_program<test_gather_flatten_stride_slice>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto x            = mm->add_parameter("X", {migraphx::shape::float_type, {1, 8}});
        auto reshape_flat = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), x);

        migraphx::shape indices_shape{migraphx::shape::int32_type, {2, 2}};
        std::vector<int32_t> indices = {1, 5, 2, 6};
        auto indices_literal         = mm->add_literal(migraphx::literal{indices_shape, indices});

        auto gather =
            mm->add_instruction(migraphx::make_op("gather"), reshape_flat, indices_literal);
        mm->add_return({gather});

        return p;
    }
};
