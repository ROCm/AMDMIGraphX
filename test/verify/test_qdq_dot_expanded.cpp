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
#include <quantize_helpers.hpp>

template <bool Transposed>
struct test_qdq_dot_expanded : verify_program<test_qdq_dot_expanded<Transposed>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* m = p.get_main_module();
        migraphx::shape ashape{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape bshape{migraphx::shape::int8_type,
                               Transposed ? std::vector<std::size_t>{3, 4}
                                          : std::vector<std::size_t>{4, 3}};
        auto a = m->add_parameter("a", ashape);
        auto b = m->add_parameter("b", bshape);
        auto sa =
            m->add_literal(migraphx::literal{{migraphx::shape::float_type, {2, 1}}, {0.25f, 0.5f}});
        auto za = m->add_literal(migraphx::literal{{migraphx::shape::int8_type, {2, 1}}, {2, -3}});
        auto qlens = Transposed ? std::vector<std::size_t>{3, 1} : std::vector<std::size_t>{1, 3};
        auto sb    = m->add_literal(
            migraphx::literal{{migraphx::shape::float_type, qlens}, {0.5f, 0.25f, 0.125f}});
        auto zb =
            m->add_literal(migraphx::literal{{migraphx::shape::int8_type, qlens}, {-2, 1, 3}});
        auto sa_bc = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ashape.lens()}}), sa);
        auto za_bc = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ashape.lens()}}), za);
        auto sb_bc = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bshape.lens()}}), sb);
        auto zb_bc = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bshape.lens()}}), zb);
        auto da = m->add_instruction(migraphx::make_op("dequantizelinear"), a, sa_bc, za_bc);
        auto db = m->add_instruction(migraphx::make_op("dequantizelinear"), b, sb_bc, zb_bc);
        if(Transposed)
            db = m->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), db);
        auto dot = m->add_instruction(migraphx::make_op("dot"), da, db);
        m->add_return({dot});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_qdq_dot_expanded<false>;
template struct test_qdq_dot_expanded<true>;
