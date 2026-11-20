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

// A recurrence over a square state whose outputs read the transposed state:
// o = output_matrix transpose(S), S = S + update_matrix update_input. Every
// shape matches the chunk scan, but
// the scan reads the state as it keeps it, so the loop must stay unrolled.
struct test_chunk_scan_transposed_state : verify_program<test_chunk_scan_transposed_state>
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
        migraphx::shape ss{migraphx::shape::float_type, {1, 2, 16, 16}};
        migraphx::shape ks{migraphx::shape::float_type, {1, 2, 2, 16, 16}};
        auto s0            = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", ks);
        auto s             = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto st = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), s);
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), st));
            auto state_update = mm->add_instruction(migraphx::make_op("dot"),
                                                    chunk(*mm, update_matrix, i),
                                                    chunk(*mm, update_input, i));
            s                 = mm->add_instruction(migraphx::make_op("add"), s, state_update);
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
        return p;
    }
};
