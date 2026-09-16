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
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>

// The second input is broadcast along the first axis and large enough that the
// fused_reduce kernel computes all the outputs along that axis in the same
// workgroup(the block_tile reduce algorithm)
template <migraphx::shape::type_t DType>
struct test_reduce_broadcast_tiled : verify_program<test_reduce_broadcast_tiled<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape as{DType, {4, 1056}};
        migraphx::shape bs{DType, {4096, 1056}};
        auto a  = mm->add_parameter("a", as);
        auto b  = mm->add_parameter("b", bs);
        auto au = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a);
        auto ab = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 4096, 1056}}}), au);
        auto bb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 4096, 1056}}}), b);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), ab, bb);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
        return p;
    };

    std::string section() const { return "reduce"; }
};

template struct test_reduce_broadcast_tiled<migraphx::shape::float_type>;
template struct test_reduce_broadcast_tiled<migraphx::shape::half_type>;
