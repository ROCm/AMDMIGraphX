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

// The unrolled delta recurrence over three chunks from a zero state: the
// first chunk emits output_base and S_1 = update_matrix update_input
// directly, the rest read the state
struct test_chunk_scan_zero_start : verify_program<test_chunk_scan_zero_start>
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
        migraphx::shape ks{migraphx::shape::float_type, {1, 2, 3, 4, 4}};
        migraphx::shape vs{migraphx::shape::float_type, {1, 2, 3, 4, 8}};
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto output_base   = mm->add_parameter("output_base", vs);
        std::vector<migraphx::instruction_ref> outs{chunk(*mm, output_base, 0)};
        auto s = mm->add_instruction(
            migraphx::make_op("dot"), chunk(*mm, update_matrix, 0), chunk(*mm, update_input, 0));
        for(std::size_t i = 1; i < 3; i++)
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
            s = mm->add_instruction(migraphx::make_op("add"), s, state_update);
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
        return p;
    }
};
