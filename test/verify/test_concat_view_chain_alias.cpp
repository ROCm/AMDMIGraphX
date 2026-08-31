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

// Concat of producers seen through transpose+reshape view chains, where
// eliminate_concat can invert the chains so the producers write directly into
// views of the concat buffer
template <migraphx::shape::type_t DType>
struct test_concat_view_chain_alias : verify_program<test_concat_view_chain_alias<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sq{DType, {1, 3, 1, 8}};
        migraphx::shape sk{DType, {1, 2, 1, 8}};
        auto q  = mm->add_parameter("q", sq);
        auto qs = mm->add_parameter("qs", sq);
        auto k  = mm->add_parameter("k", sk);
        auto ks = mm->add_parameter("ks", sk);
        auto q2 = mm->add_instruction(migraphx::make_op("mul"), q, qs);
        auto k2 = mm->add_instruction(migraphx::make_op("mul"), k, ks);
        auto tq = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), q2);
        auto rq = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 24}}}), tq);
        auto tk = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), k2);
        auto rk = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 16}}}), tk);
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), rq, rk);
        return p;
    }
};

template struct test_concat_view_chain_alias<migraphx::shape::float_type>;
template struct test_concat_view_chain_alias<migraphx::shape::half_type>;
