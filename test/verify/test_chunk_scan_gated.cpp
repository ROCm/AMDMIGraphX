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
// u = update_input - read_matrix S, o = output_base + output_matrix S,
// S = S * decay + update_matrix u. The gpu target walks the chunks in one
// kernel, accumulating half operands in float; 8 rows per chunk make the
// stacked output and read matrix block of a matrix instruction span both.
template <migraphx::shape::type_t DType>
struct test_chunk_scan_gated : verify_program<test_chunk_scan_gated<DType>>
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
        migraphx::shape ss{DType, {1, 2, 16, 16}};
        auto s0            = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", {DType, {1, 2, 2, 8, 16}});
        auto output_matrix = mm->add_parameter("output_matrix", {DType, {1, 2, 2, 8, 16}});
        auto update_matrix = mm->add_parameter("update_matrix", {DType, {1, 2, 2, 16, 8}});
        auto update_input  = mm->add_parameter("update_input", {DType, {1, 2, 2, 8, 16}});
        auto output_base   = mm->add_parameter("output_base", {DType, {1, 2, 2, 8, 16}});
        auto decay         = mm->add_parameter("decay", {DType, {1, 2, 2, 1, 1}});
        auto s             = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto u = mm->add_instruction(
                migraphx::make_op("sub"), chunk(*mm, update_input, i), read_product);
            auto output_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
            outs.push_back(mm->add_instruction(
                migraphx::make_op("add"), chunk(*mm, output_base, i), output_product));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            auto decay_i =
                mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}),
                                    chunk(*mm, decay, i));
            auto sg = mm->add_instruction(migraphx::make_op("mul"), s, decay_i);
            s       = mm->add_instruction(migraphx::make_op("add"), sg, state_update);
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
        return p;
    }
};

template struct test_chunk_scan_gated<migraphx::shape::float_type>;
template struct test_chunk_scan_gated<migraphx::shape::half_type>;
