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
 *
 */
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/hoist_kv_cache_attention_masks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <group.hpp>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::hoist_kv_cache_attention_masks{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(hoist_decode_mask)
{
    migraphx::shape qs{migraphx::shape::half_type, {1, 2, 1, 2}};
    migraphx::shape kvs{migraphx::shape::half_type, {1, 2, 4, 2}};
    migraphx::shape ss{migraphx::shape::int32_type, {1, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto q   = mm->add_parameter("q", qs);
        auto k   = mm->add_parameter("k", kvs);
        auto v   = mm->add_parameter("v", kvs);
        auto slk = mm->add_parameter("slk", ss);
        auto group =
            add_group(p1, "attn0", "kv_cache_attention", {q, k, v, slk}, [](auto* gm, auto params) {
                auto range = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 2, 3, 4}});
                auto bc    = gm->add_instruction(
                    migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4}}}), range);
                auto past = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), params[3]);
                auto gt   = gm->add_instruction(migraphx::make_op("greater"), bc, past);
                auto mask = gm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
                    gt);
                auto mask4 =
                    gm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), mask);
                auto maskb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), mask4);
                auto kt = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), params[1]);
                auto scores = gm->add_instruction(migraphx::make_op("dot"), params[0], kt);
                auto ninf   = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::half_type, {1}}, {-65504.0f}});
                auto ninfb  = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), ninf);
                auto masked = gm->add_instruction(migraphx::make_op("where"), maskb, ninfb, scores);
                auto sm  = gm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), masked);
                auto out = gm->add_instruction(migraphx::make_op("dot"), sm, params[2]);
                return std::vector<migraphx::instruction_ref>{out};
            });
        mm->add_return({group});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto q     = mm->add_parameter("q", qs);
        auto k     = mm->add_parameter("k", kvs);
        auto v     = mm->add_parameter("v", kvs);
        auto slk   = mm->add_parameter("slk", ss);
        auto range = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 2, 3, 4}});
        auto bc = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4}}}), range);
        auto past =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), slk);
        auto gt   = mm->add_instruction(migraphx::make_op("greater"), bc, past);
        auto mask = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto group = add_group(
            p2, "attn0", "kv_cache_attention", {mask, k, q, v}, [](auto* gm, auto params) {
                auto mask4 = gm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}),
                                                 params[0]);
                auto maskb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), mask4);
                auto kt = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), params[1]);
                auto scores = gm->add_instruction(migraphx::make_op("dot"), params[2], kt);
                auto ninf   = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::half_type, {1}}, {-65504.0f}});
                auto ninfb  = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), ninf);
                auto masked = gm->add_instruction(migraphx::make_op("where"), maskb, ninfb, scores);
                auto sm  = gm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), masked);
                auto out = gm->add_instruction(migraphx::make_op("dot"), sm, params[3]);
                return std::vector<migraphx::instruction_ref>{out};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());

    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(hoist_prefill_mask_drops_unused_input)
{
    migraphx::shape qs{migraphx::shape::half_type, {1, 2, 2, 2}};
    migraphx::shape kvs{migraphx::shape::half_type, {1, 2, 4, 2}};
    migraphx::shape ss{migraphx::shape::int32_type, {1, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto q   = mm->add_parameter("q", qs);
        auto k   = mm->add_parameter("k", kvs);
        auto v   = mm->add_parameter("v", kvs);
        auto slk = mm->add_parameter("slk", ss);
        auto group =
            add_group(p1, "attn0", "kv_cache_attention", {q, k, v, slk}, [](auto* gm, auto params) {
                auto range  = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::int32_type, {1, 4}}, {0, 1, 2, 3}});
                auto rangeb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), range);
                auto rows = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::int32_type, {2, 1}}, {0, 1}});
                auto past = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), params[3]);
                auto pos   = gm->add_instruction(migraphx::make_op("add"), rows, past);
                auto limit = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::int32_type, {2, 1}}, {3, 3}});
                auto last  = gm->add_instruction(migraphx::make_op("min"), pos, limit);
                auto lastb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), last);
                auto gt   = gm->add_instruction(migraphx::make_op("greater"), rangeb, lastb);
                auto mask = gm->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
                    gt);
                auto mask4 =
                    gm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), mask);
                auto maskb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4}}}), mask4);
                auto kt = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), params[1]);
                auto scores = gm->add_instruction(migraphx::make_op("dot"), params[0], kt);
                auto ninf   = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::half_type, {1}}, {-65504.0f}});
                auto ninfb  = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4}}}), ninf);
                auto masked = gm->add_instruction(migraphx::make_op("where"), maskb, ninfb, scores);
                auto sm  = gm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), masked);
                auto out = gm->add_instruction(migraphx::make_op("dot"), sm, params[2]);
                return std::vector<migraphx::instruction_ref>{out};
            });
        mm->add_return({group});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto q     = mm->add_parameter("q", qs);
        auto k     = mm->add_parameter("k", kvs);
        auto v     = mm->add_parameter("v", kvs);
        auto slk   = mm->add_parameter("slk", ss);
        auto range = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {1, 4}}, {0, 1, 2, 3}});
        auto rangeb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), range);
        auto rows = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2, 1}}, {0, 1}});
        auto past =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), slk);
        auto pos   = mm->add_instruction(migraphx::make_op("add"), rows, past);
        auto limit = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {2, 1}}, {3, 3}});
        auto last = mm->add_instruction(migraphx::make_op("min"), pos, limit);
        auto lastb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), last);
        auto gt   = mm->add_instruction(migraphx::make_op("greater"), rangeb, lastb);
        auto mask = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto group = add_group(
            p2, "attn0", "kv_cache_attention", {mask, k, q, v}, [](auto* gm, auto params) {
                auto mask4 = gm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}),
                                                 params[0]);
                auto maskb = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4}}}), mask4);
                auto kt = gm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), params[1]);
                auto scores = gm->add_instruction(migraphx::make_op("dot"), params[2], kt);
                auto ninf   = gm->add_literal(migraphx::literal{
                    migraphx::shape{migraphx::shape::half_type, {1}}, {-65504.0f}});
                auto ninfb  = gm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4}}}), ninf);
                auto masked = gm->add_instruction(migraphx::make_op("where"), maskb, ninfb, scores);
                auto sm  = gm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), masked);
                auto out = gm->add_instruction(migraphx::make_op("dot"), sm, params[3]);
                return std::vector<migraphx::instruction_ref>{out};
            });
        mm->add_return({group});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
