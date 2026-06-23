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

// Regression for the historical GPU JIT NonZero kernel that used a uint8_t
// accumulator inside block_scan: once the per-block set-bit count rolled past
// 255 the prefix-sum wrapped and the kernel produced out-of-bounds writes,
// silently corrupting downstream consumers (notably ScatterND in the
// vision_encoder pipeline). With 32x32 = 1024 elements and a random fill that
// puts most positions nonzero (every bool and float verify_program input is
// generated densely), the kernel comfortably crosses the historical 255-set-
// bit limit on every run.
template <migraphx::shape::type_t DType>
struct test_nonzero_large : verify_program<test_nonzero_large<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {32, 32}};
        auto x = mm->add_parameter("data", s);
        auto r = mm->add_instruction(migraphx::make_op("nonzero"), x);
        mm->add_return({r});

        return p;
    }
};

template struct test_nonzero_large<migraphx::shape::bool_type>;
template struct test_nonzero_large<migraphx::shape::float_type>;
