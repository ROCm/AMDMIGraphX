/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

/**
 * Test that the split_single_dyn_dim GPU compiler pass produces the same results as ref.
 */
struct test_split_single_dyn_dim : verify_program<test_split_single_dyn_dim>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input = mm->add_parameter("data", s);
        auto broadcast_lit =
            mm->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input);
        auto add_ins = mm->add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        mm->add_return({add_ins});
        return p;
    }
};
