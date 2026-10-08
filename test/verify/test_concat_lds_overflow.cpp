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
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <string>

// The block_tile concat algorithm tiles NGroups * ninputs * max_size elements
// into LDS; with enough inputs that exceeds the 64KB workgroup limit
template <migraphx::shape::type_t DType, std::size_t N, std::size_t Rows, std::size_t Width>
struct test_concat_lds_overflow : verify_program<test_concat_lds_overflow<DType, N, Rows, Width>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {Rows, Width}};
        std::vector<migraphx::instruction_ref> args;
        auto r = migraphx::range(N);
        std::transform(r.begin(), r.end(), std::back_inserter(args), [&](auto i) {
            return mm->add_parameter("x" + std::to_string(i), s);
        });
        mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}), args);
        return p;
    }
};

// 16 groups * 30 inputs * 60 elements * 4 bytes = 115200 bytes of LDS
template struct test_concat_lds_overflow<migraphx::shape::float_type, 30, 16, 60>;
