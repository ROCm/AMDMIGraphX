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
#include <migraphx/split_seq_len.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::split_seq_len{}, migraphx::dead_code_elimination{}});
}

static const migraphx::sym::expr& seq_var()
{
    static const auto seq = migraphx::sym::var("seq", {1, 4});
    return seq;
}

// x {1, seq, 4} -> y = x + 1; the kv cache holds y rows: a minimal kv-cache model
static migraphx::program make_kv_cache_program()
{
    using migraphx::sym::lit;
    migraphx::program p;
    auto* mm = p.get_main_module();
    const migraphx::shape x_s{migraphx::shape::float_type,
                              {migraphx::shape::dynamic_dimension{lit(std::int64_t{1})},
                               migraphx::shape::dynamic_dimension{seq_var()},
                               migraphx::shape::dynamic_dimension{lit(std::int64_t{4})}}};
    auto x    = mm->add_parameter("x", x_s);
    auto slk  = mm->add_parameter("slk", {migraphx::shape::int32_type, {1}});
    auto past = mm->add_parameter("past", {migraphx::shape::float_type, {1, 1, 4, 4}});
    auto one =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1}});
    auto oneb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(x_s.dyn_dims())}}),
        one,
        x);
    auto y   = mm->add_instruction(migraphx::make_op("add"), x, oneb);
    auto cur = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), y);
    auto k   = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 1}}), cur, slk, past);
    mm->add_return({y, k});
    return p;
}

TEST_CASE(split_kv_cache_seq_len)
{
    using migraphx::sym::lit;
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();
        const migraphx::shape x_s{migraphx::shape::float_type,
                                  {migraphx::shape::dynamic_dimension{lit(std::int64_t{1})},
                                   migraphx::shape::dynamic_dimension{seq_var()},
                                   migraphx::shape::dynamic_dimension{lit(std::int64_t{4})}}};
        const migraphx::shape slk_s{migraphx::shape::int32_type, {1}};
        const migraphx::shape past_s{migraphx::shape::float_type, {1, 1, 4, 4}};

        // Submodule parameters are named by select_module argument position: the shared
        // past and slk, the padded copy of x, then x itself (decode only)
        const migraphx::shape padded_s{migraphx::shape::float_type, {1, 4, 4}};
        auto create_submodule = [&](std::size_t seq_len, bool padded) {
            auto* submod = p0.create_module("seq_len_" + std::to_string(seq_len));
            auto sm_past = submod->add_parameter("x0", past_s);
            auto sm_slk  = submod->add_parameter("x1", slk_s);
            auto sm_x    = submod->add_parameter("x2", padded_s);
            if(not padded)
                sm_x = submod->add_parameter(
                    "x3", migraphx::shape{migraphx::shape::float_type, {1, seq_len, 4}});
            auto sm_one = submod->add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {1}});
            auto sm_oneb = submod->add_instruction(
                migraphx::make_op("multibroadcast",
                                  {{"out_dyn_dims", migraphx::to_value(x_s.dyn_dims())}}),
                sm_one,
                sm_x);
            auto sm_y = submod->add_instruction(migraphx::make_op("add"), sm_x, sm_oneb);
            auto sm_cur =
                submod->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), sm_y);
            auto sm_k = submod->add_instruction(
                migraphx::make_op("concat_past_present", {{"kv_num_heads", 1}}),
                sm_cur,
                sm_slk,
                sm_past);
            submod->add_return({sm_y, sm_k});
            return submod;
        };

        auto x      = mm0->add_parameter("x", x_s);
        auto slk    = mm0->add_parameter("slk", slk_s);
        auto past   = mm0->add_parameter("past", past_s);
        auto* mod1  = create_submodule(1, false);
        auto* mod4  = create_submodule(4, true);
        auto padded = mm0->add_instruction(migraphx::make_op("fixed_pad"), x);
        const migraphx::shape out_attr{{x_s, past_s}};
        auto sm_ins = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {past, slk, padded, x},
            {mod1, mod4});
        auto gte0 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        auto ends = mm0->add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{seq_var()})}}),
            x);
        auto starts = mm0->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
        auto trimmed = mm0->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts",
                  migraphx::to_value(std::vector<migraphx::sym::expr>{lit(std::int64_t{0})})},
                 {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{seq_var()})}}),
            gte0,
            starts,
            ends);
        auto gte1 =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sm_ins);
        mm0->add_return({trimmed, gte1});
    }

    auto p1 = make_kv_cache_program();
    run_pass(p1);

    EXPECT(p0 == p1);
}

TEST_CASE(split_seq_len_no_kv_cache)
{
    using migraphx::sym::lit;
    // No concat_past_present, so the pass should not apply
    migraphx::program p;
    auto* mm = p.get_main_module();
    const migraphx::shape x_s{migraphx::shape::float_type,
                              {migraphx::shape::dynamic_dimension{lit(std::int64_t{1})},
                               migraphx::shape::dynamic_dimension{seq_var()}}};
    auto x = mm->add_parameter("x", x_s);
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, x)});
    auto expected = p;
    run_pass(p);
    EXPECT(expected == p);
}

TEST_CASE(split_seq_len_static_params)
{
    // No symbolic sequence dimension, so the pass should not apply
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 4}});
    auto slk  = mm->add_parameter("slk", {migraphx::shape::int32_type, {1}});
    auto past = mm->add_parameter("past", {migraphx::shape::float_type, {1, 1, 4, 4}});
    auto cur  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
    auto k    = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 1}}), cur, slk, past);
    mm->add_return({k});
    auto expected = p;
    run_pass(p);
    EXPECT(expected == p);
}

TEST_CASE(split_seq_len_ref_eval)
{
    // Decode (seq=1) runs the exact submodule; any longer prompt runs zero-padded at the
    // maximum and the trimmed output matches the unpadded computation.
    for(const std::size_t seq_len : {std::size_t{1}, std::size_t{2}, std::size_t{4}})
    {
        auto p = make_kv_cache_program();
        // The pipeline staticizes the cloned submodules with simplify_dyn_ops
        migraphx::run_passes(p,
                             {migraphx::split_seq_len{},
                              migraphx::dead_code_elimination{},
                              migraphx::simplify_dyn_ops{},
                              migraphx::dead_code_elimination{}});
        p.compile(migraphx::make_target("ref"));

        std::vector<float> x_data(seq_len * 4);
        std::iota(x_data.begin(), x_data.end(), 1.0f);
        std::vector<int32_t> slk_data{int32_t(seq_len) - 1};
        std::vector<float> past_data(16, -1.0f);

        migraphx::parameter_map params;
        params["x"] =
            migraphx::argument({migraphx::shape::float_type, {1, seq_len, 4}}, x_data.data());
        params["slk"] = migraphx::argument({migraphx::shape::int32_type, {1}}, slk_data.data());
        params["past"] =
            migraphx::argument({migraphx::shape::float_type, {1, 1, 4, 4}}, past_data.data());
        auto results = p.eval(params);
        EXPECT(results.size() == 2);

        // y = x + 1, trimmed back to the actual sequence length
        std::vector<float> y;
        results.at(0).visit([&](auto output) { y.assign(output.begin(), output.end()); });
        EXPECT(results.at(0).get_shape().lens() == std::vector<std::size_t>({1, seq_len, 4}));
        std::vector<float> y_gold(seq_len * 4);
        std::transform(x_data.begin(), x_data.end(), y_gold.begin(), [](auto v) { return v + 1; });
        EXPECT(migraphx::verify::verify_rms_range(y, y_gold));

        // The cache rows for the current tokens hold y; a padded prompt writes the
        // computation of the zero padding (0 + 1) past the actual length, and decode
        // leaves those rows untouched
        std::vector<float> k;
        results.at(1).visit([&](auto output) { k.assign(output.begin(), output.end()); });
        std::vector<float> k_gold(16, seq_len == 1 ? -1.0f : 1.0f);
        std::copy(y_gold.begin(), y_gold.end(), k_gold.begin());
        EXPECT(migraphx::verify::verify_rms_range(k, k_gold));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
