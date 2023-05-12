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
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_gathernd_1d : verify_program<test_gathernd_1d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape ds{migraphx::shape::float_type, {90000, 264}};
        auto a0 = mm->add_parameter("data", ds);

        migraphx::shape gs{migraphx::shape::float_type, {1}, {0}};
        auto a3 = mm->add_literal(migraphx::literal{gs, {0.5}});

        migraphx::shape neg1{migraphx::shape::int64_type, {1}, {1}};
        auto lneg1 = mm->add_literal(migraphx::literal{neg1, {-1}});

        //std::vector<int64_t> indices(23670000, 0);
        //migraphx::shape is{migraphx::shape::int64_type, {23670000, 1}};
        //auto a1 = mm->add_literal(migraphx::literal{is, indices});

        auto sig_out = mm->add_instruction(migraphx::make_op("sigmoid"), a0);
        //auto sig_sh  = sig_out.get_shape();
        //auto sig_lit = mm->add_literal(migraphx::literal{sig_sh, sig_sh->lget_shape().lens()});

        //auto slice_o = mm->add_instruction(migraphx::make_op("slice"), sig_lit); //shape operator
        //auto concat_o = mm->add_instruction(migraphx::make_op("concat"), slice_o, lneg1);

        auto con_out = mm->add_instruction(migraphx::make_op("contiguous"), sig_out);
        auto mult_b  = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {23760000}}}), a3);

        p.debug_print();
        auto re_out  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", -1}}), con_out);
        auto g_out   = mm->add_instruction(migraphx::make_op("greater"), sig_out, mult_b);
        auto conv_o  = mm->add_instruction(migraphx::make_op("convert", {{"target_type", 0}}), g_out);
        auto nonzout = mm->add_instruction(migraphx::make_op("nonzero"), conv_o);
        auto trans_o = mm->add_instruction(migraphx::make_op("transpose"), nonzout);

        p.debug_print();
        mm->add_instruction(migraphx::make_op("gathernd"), sig_out, trans_o);
        return p;
    }
};
