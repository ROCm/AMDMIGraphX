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

// kv-cache append whose producer is a bias add in the flat (batch, seq, heads*head)
// projection layout; the gpu target moves the reshape and transpose onto the
// add's inputs so it writes the cache slot directly
template <std::size_t Seq>
struct test_concat_past_present_add_flat : verify_program<test_concat_past_present_add_flat<Seq>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {1, Seq, 8}};
        migraphx::shape cs{migraphx::shape::half_type, {1, 2, 8, 4}};
        auto x     = mm->add_parameter("x", s);
        auto bias  = mm->add_parameter("bias", {migraphx::shape::half_type, {8}});
        auto cache = mm->add_parameter("cache", cs);
        auto slk   = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 1}}, {Seq}});
        auto bb = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}),
                                      bias);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, bb);
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, Seq, 2, 4}}}), add);
        auto trp = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsp);
        mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), trp, slk, cache);
        return p;
    }
};

template struct test_concat_past_present_add_flat<1>;
template struct test_concat_past_present_add_flat<4>;
