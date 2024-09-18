/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
    migraphx::shape qkvs{migraphx::shape::half_type, {1, 1, 12288}};
    migraphx::shape pkvs{migraphx::shape::half_type, {1, 32, 4096, 128}};
    migraphx::shape kvs{migraphx::shape::float_type, {1}};
    migraphx::shape consts{migraphx::shape::int32_type, {1}};
    migraphx::shape cs{migraphx::shape::half_type, {4096, 64}};
    migraphx::shape outs{migraphx::shape::half_type, {1, 1, 4096}};

    std::vector<float> qkv_data(qkvs.elements(), 1.0);
    std::vector<float> pkv_data(pkvs.elements(), 0.0);
    std::vector<float> cs_data(cs.elements(), 1.0);
    auto qkv = mm->add_literal(migraphx::literal{qkvs, qkv_data});
    auto kv  = mm->add_literal(migraphx::literal{kvs, {1}});
    auto pk  = mm->add_literal(migraphx::literal{pkvs, pkv_data});
    auto pv  = mm->add_literal(migraphx::literal{pkvs, pkv_data});
    auto slk = mm->add_literal(migraphx::literal{consts, {0}});
    auto tsl = mm->add_literal(migraphx::literal{consts, {1}});
    auto cc  = mm->add_literal(migraphx::literal{cs, cs_data});
    auto sc  = mm->add_literal(migraphx::literal{cs, cs_data});

    auto gqa        = mm->add_instruction(migraphx::make_op("group_query_attention",
                                                            {{"do_rotary", 1},
                                                             {"kv_num_heads", 32},
                                                             {"local_window_size", -1},
                                                             {"num_heads", 32},
                                                             {"rotary_interleaved", 0}}),
                                   qkv,
                                   kv,
                                   kv,
                                   pk,
                                   pv,
                                   slk,
                                   tsl,
                                   cc,
                                   sc);
    auto gqa_output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), gqa);
    auto gqa_present_key =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), gqa);
    auto gqa_present_value =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), gqa);
    mm->add_return({gqa_output, gqa_present_key, gqa_present_value});
    p.compile(migraphx::make_target("ref"));
    auto outputs = p.eval({});

    auto result = outputs.front();
    std::vector<float> results_vector(outs.elements());
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    auto pres_key = outputs.at(1);
    std::vector<float> pres_key_vector(pkvs.elements());
    pres_key.visit([&](auto output) { pres_key_vector.assign(output.begin(), output.end()); });
    auto pres_val = outputs.back();
    std::vector<float> pres_val_vector(pkvs.elements());
    pres_val.visit([&](auto output) { pres_val_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_output(outs.elements(), 1.0);
    std::vector<float> gold_k_cache(pkvs.elements(), 0.0);
    std::vector<float> gold_v_cache(pkvs.elements(), 0.0);
    auto kv_cache_lens = pkvs.lens();
    for(auto i = 0; i < pkvs.elements(); i += kv_cache_lens[2] * kv_cache_lens[3])
    {
        for(auto j = 0; j < kv_cache_lens[3]; j++)
        {
            gold_k_cache[i + j] = j >= 64 ? 2.0 : 0.0;
            gold_v_cache[i + j] = 1.0;
        }
    }
    EXPECT(results_vector == gold_output);
    EXPECT(pres_key_vector == gold_k_cache);
    EXPECT(pres_val_vector == gold_v_cache);
}
