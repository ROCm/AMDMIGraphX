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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_flash_decode.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <numeric>
#include <test.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::gpu::fuse_flash_decode{}, migraphx::dead_code_elimination{}});
}

static bool contains_op(const migraphx::module& m, const std::string& name)
{
    return std::any_of(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == name; });
}

// Build the kv-cache decode attention group the way find_kv_cache_attention
// produces it for qwen-style GQA models
static migraphx::program
make_kv_cache_attention(int64_t b, int64_t qh, int64_t kvh, int64_t d, int64_t n)
{
    const int64_t heads = qh + kvh;
    const int64_t ratio = qh / kvh;
    migraphx::shape qk_s{migraphx::shape::bf16_type,
                         {std::size_t(b), std::size_t(heads), 1, std::size_t(d)}};
    migraphx::shape kv_s{migraphx::shape::bf16_type,
                         {std::size_t(b), std::size_t(kvh), std::size_t(n), std::size_t(d)}};
    migraphx::shape sl_s{migraphx::shape::int32_type, {std::size_t(b), 1}};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto qk  = mm->add_parameter("qk", qk_s);
    auto v   = mm->add_parameter("v", kv_s);
    auto k   = mm->add_parameter("k", kv_s);
    auto sl  = mm->add_parameter("sl", sl_s);

    auto* am = p.create_module("attn0");
    am->set_bypass();
    auto x0 = am->add_parameter("x0", qk_s);
    auto x1 = am->add_parameter("x1", kv_s);
    auto x2 = am->add_parameter("x2", kv_s);
    auto x3 = am->add_parameter("x3", sl_s);

    auto q = am->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {qh}}}), x0);

    auto gqa_expand = [&](migraphx::instruction_ref in, bool transpose) {
        auto u = am->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), in);
        if(transpose)
            u = am->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), u);
        auto lens = u->get_shape().lens();
        lens[2]   = ratio;
        auto mb = am->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), u);
        std::vector<int64_t> dims = {b, qh};
        dims.push_back(lens[3]);
        dims.push_back(lens[4]);
        return am->add_instruction(migraphx::make_op("reshape", {{"dims", dims}}), mb);
    };
    auto kt = gqa_expand(x2, true);
    auto vv = gqa_expand(x1, false);

    auto dot1 = am->add_instruction(migraphx::make_op("dot"), q, kt);
    auto sc   = am->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::bf16_type, {1}, {1}}, {0.125f}});
    auto scb = am->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), sc);
    auto s_f = am->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot1);
    auto sc_f = am->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scb);
    auto mul = am->add_instruction(migraphx::make_op("mul"), s_f, sc_f);

    std::vector<int> range_vec(n);
    std::iota(range_vec.begin(), range_vec.end(), 0);
    auto range   = am->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::int32_type, {std::size_t(n)}}, range_vec});
    auto range_b = am->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {b, n}}}), range);
    auto sl_b =
        am->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {b, n}}}), x3);
    auto gt  = am->add_instruction(migraphx::make_op("greater"), range_b, sl_b);
    auto gtb = am->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
    auto gtu  = am->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), gtb);
    auto mask = am->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", mul->get_shape().lens()}}), gtu);

    auto ninf =
        am->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::bf16_type, {1}, {1}},
                                          {-std::numeric_limits<float>::infinity()}});
    auto ninfb = am->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", mul->get_shape().lens()}}), ninf);
    auto ninf_f = am->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ninfb);

    auto where = am->add_instruction(migraphx::make_op("where"), mask, ninf_f, mul);
    auto rmax  = am->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
    auto rmaxb = am->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", where->get_shape().lens()}}), rmax);
    auto sub   = am->add_instruction(migraphx::make_op("sub"), where, rmaxb);
    auto ex    = am->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum  = am->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), ex);
    auto rsumb = am->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", ex->get_shape().lens()}}), rsum);
    auto div  = am->add_instruction(migraphx::make_op("div"), ex, rsumb);
    auto prob = am->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), div);
    auto dot2 = am->add_instruction(migraphx::make_op("dot"), prob, vv);
    auto tp =
        am->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), dot2);
    auto out = am->add_instruction(migraphx::make_op("reshape", {{"dims", {b, 1, qh * d}}}), tp);
    am->add_return({out});

    auto group = mm->add_instruction(
        migraphx::make_op("group", {{"tag", "kv_cache_attention"}}), {qk, v, k, sl}, {am});
    mm->add_return({group});
    return p;
}

TEST_CASE(fuse_flash_decode_long_context)
{
    auto p = make_kv_cache_attention(2, 4, 2, 64, 2048);
    run_pass(p);
    const auto* mm = p.get_main_module();
    EXPECT(contains_op(*mm, "gpu::kv_flash_decode_splitk"));
    EXPECT(contains_op(*mm, "gpu::kv_flash_decode_reduce"));
    EXPECT(not contains_op(*mm, "group"));

    auto splitk = std::find_if(mm->begin(), mm->end(), [](const auto& ins) {
        return ins.name() == "gpu::kv_flash_decode_splitk";
    });
    auto v      = splitk->get_operator().to_value();
    EXPECT(v.at("q_heads").to<std::size_t>() == 4);
    EXPECT(v.at("kv_heads").to<std::size_t>() == 2);
    EXPECT(migraphx::float_equal(v.at("scale").to<float>(), 0.125f));
}

// Short caches stay on the fused attention kernel
TEST_CASE(fuse_flash_decode_skips_short_context)
{
    auto p = make_kv_cache_attention(2, 4, 2, 64, 512);
    run_pass(p);
    EXPECT(not contains_op(*p.get_main_module(), "gpu::kv_flash_decode_splitk"));
    EXPECT(contains_op(*p.get_main_module(), "group"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
