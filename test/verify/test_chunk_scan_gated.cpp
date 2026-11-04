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

// The unrolled gated delta recurrence over two chunks with a past state:
// u = u0 - w S, o = pu0 + q S, S = S * gamma + kt u. The gpu target walks
// the chunks in one kernel.
struct test_chunk_scan_gated : verify_program<test_chunk_scan_gated>
{
    static migraphx::instruction_ref
    chunk(migraphx::module& m, migraphx::instruction_ref x, std::size_t i)
    {
        auto s = m.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {i}}, {"ends", {i + 1}}}), x);
        return m.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), s);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape ss{migraphx::shape::float_type, {1, 2, 4, 8}};
        auto s0  = mm->add_parameter("s0", ss);
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {1, 2, 2, 4, 4}});
        auto q   = mm->add_parameter("q", {migraphx::shape::float_type, {1, 2, 2, 4, 4}});
        auto kt  = mm->add_parameter("kt", {migraphx::shape::float_type, {1, 2, 2, 4, 4}});
        auto u0  = mm->add_parameter("u0", {migraphx::shape::float_type, {1, 2, 2, 4, 8}});
        auto pu0 = mm->add_parameter("pu0", {migraphx::shape::float_type, {1, 2, 2, 4, 8}});
        auto g   = mm->add_parameter("g", {migraphx::shape::float_type, {1, 2, 2, 1, 1}});
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto ws = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, w, i), s);
            auto u  = mm->add_instruction(migraphx::make_op("sub"), chunk(*mm, u0, i), ws);
            auto qs = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s);
            outs.push_back(mm->add_instruction(migraphx::make_op("add"), chunk(*mm, pu0, i), qs));
            auto upd = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, kt, i), u);
            auto gi  = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}), chunk(*mm, g, i));
            auto sg = mm->add_instruction(migraphx::make_op("mul"), s, gi);
            s       = mm->add_instruction(migraphx::make_op("add"), sg, upd);
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
        return p;
    }
};
