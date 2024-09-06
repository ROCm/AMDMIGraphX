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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(gqa_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape qkvs{migraphx::shape::half_type, {1, 64, 12288}};
    migraphx::shape pkvs{migraphx::shape::half_type, {1, 32, 4096, 128}};
    migraphx::shape consts{migraphx::shape::int32_type, {1}};
    migraphx::shape cs{migraphx::shape::half_type, {4096, 64}};
    migraphx::shape outs{migraphx::shape::half_type, {1, 64, 4096}};

    std::vector<float> qkv_data(qkvs.elements(), 1.0);
    std::vector<float> pkv_data(pkvs.elements(), 0.0);
    std::vector<float> cs_data(cs.elements(), 1.0);
    auto qkv = mm->add_literal(migraphx::literal{qkvs, qkv_data});
    auto pk  = mm->add_literal(migraphx::literal{pkvs, pkv_data});
    auto pv  = mm->add_literal(migraphx::literal{pkvs, pkv_data});
    auto slk = mm->add_literal(migraphx::literal{consts, {0}});
    auto tsl = mm->add_literal(migraphx::literal{consts, {64}});
    auto cc  = mm->add_literal(migraphx::literal{cs, cs_data});
    auto sc  = mm->add_literal(migraphx::literal{cs, cs_data});

    mm->add_instruction(migraphx::make_op("group_query_attention",
                                          {{"do_rotary", 1},
                                           {"kv_num_heads", 32},
                                           {"local_window_size", -1},
                                           {"num_heads", 32},
                                           {"rotary_interleaved", 0}}),
                        qkv,
                        pk,
                        pv,
                        slk,
                        tsl,
                        cc,
                        sc);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(outs.elements());
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    for(auto i = 0; i < outs.elements(); ++i)
    {
        std::cout << results_vector[i] << std::endl;
    }
}