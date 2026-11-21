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
#include <test.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/target.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

// fp16 attention shaped like the dolly-v2-7b export: a 3-D q.k^T gemm unsqueezed to 4-D,
// scaled by 1/sqrt(d), causal where, softmax, then the second gemm. The block is too small
// for fuse_attention to fuse it, so q.k^T runs as its own gemm. With q = k = 24 every
// unscaled score is 24 * 24 * 128 = 73728, past the fp16 max of 65504, while the scaled
// score (6517) fits. All scores in a row are equal, so the causal softmax is uniform and
// output row i is the mean of v rows 0..i. With v row s filled with s, that is i / 2.
TEST_CASE(unfused_attention_fp16_scores_do_not_overflow)
{
    const std::size_t heads = 32;
    const std::size_t seq   = 8;
    const std::size_t dim   = 128;
    migraphx::shape qk_s{migraphx::shape::half_type, {heads, seq, dim}};
    migraphx::shape v_s{migraphx::shape::half_type, {1, heads, seq, dim}};
    std::vector<std::size_t> scores_lens = {1, heads, seq, seq};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", qk_s);
    auto k   = mm->add_parameter("k", qk_s);
    auto v   = mm->add_parameter("v", v_s);
    auto kt  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
    auto qk  = mm->add_instruction(migraphx::make_op("dot"), q, kt);
    auto scores = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), qk);
    auto scale  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {1.0 / std::sqrt(dim)}});
    scale  = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}),
                                 scale);
    scores = mm->add_instruction(migraphx::make_op("mul"), scores, scale);
    migraphx::shape mask_s{migraphx::shape::bool_type, {1, 1, seq, seq}};
    std::vector<char> causal(mask_s.elements());
    migraphx::shape_for_each(
        mask_s, [&](const auto& idx) { causal[mask_s.index(idx)] = idx[3] <= idx[2]; });
    auto mask = mm->add_literal(migraphx::literal{mask_s, causal});
    mask =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}), mask);
    auto lowest =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-65504.0}});
    lowest = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}),
                                 lowest);
    scores = mm->add_instruction(migraphx::make_op("where"), mask, scores, lowest);
    auto probs = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), scores);
    auto out   = mm->add_instruction(migraphx::make_op("dot"), probs, v);
    mm->add_return({out});

    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);

    // v row s holds s, so output row i is i / 2
    std::vector<float> qk_data(qk_s.elements(), 24.0f);
    std::vector<float> v_data(v_s.elements());
    std::vector<float> expected(v_s.elements());
    migraphx::shape_for_each(v_s, [&](const auto& idx) {
        v_data[v_s.index(idx)]   = idx[2];
        expected[v_s.index(idx)] = idx[2] / 2.0f;
    });
    auto q_arg  = migraphx::literal{qk_s, qk_data}.get_argument();
    auto k_arg  = migraphx::literal{qk_s, qk_data}.get_argument();
    auto v_arg  = migraphx::literal{v_s, v_data}.get_argument();
    auto result = p.eval({{"q", q_arg}, {"k", k_arg}, {"v", v_arg}}).back();

    std::vector<float> out_data;
    result.visit([&](auto x) { out_data.assign(x.begin(), x.end()); });
    EXPECT(std::equal(
        out_data.begin(), out_data.end(), expected.begin(), expected.end(), [](float x, float y) {
            return std::abs(x - y) < 1e-2f;
        }));
}

// The same for a decode step: the new key and value appended to the cache with a concat and one
// q row per head, which is also too small to fuse. All scores are equal, so the output is the
// mean of the v rows: with v row s filled with s, (kv - 1) / 2.
TEST_CASE(unfused_decode_attention_fp16_scores_do_not_overflow)
{
    const std::size_t heads = 32;
    const std::size_t past  = 24;
    const std::size_t kv    = past + 1;
    const std::size_t dim   = 128;
    migraphx::shape q_s{migraphx::shape::half_type, {heads, 1, dim}};
    migraphx::shape new_s{migraphx::shape::half_type, {1, heads, 1, dim}};
    migraphx::shape past_s{migraphx::shape::half_type, {1, heads, past, dim}};
    std::vector<std::size_t> scores_lens = {1, heads, 1, kv};

    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto q      = mm->add_parameter("q", q_s);
    auto past_k = mm->add_parameter("past_k", past_s);
    auto k      = mm->add_parameter("k", new_s);
    auto past_v = mm->add_parameter("past_v", past_s);
    auto v      = mm->add_parameter("v", new_s);
    auto kc     = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), past_k, k);
    auto vc     = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), past_v, v);
    auto kt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), kc);
    kt          = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), kt);
    auto qk     = mm->add_instruction(migraphx::make_op("dot"), q, kt);
    auto scores = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), qk);
    auto scale  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {1.0 / std::sqrt(dim)}});
    scale  = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}),
                                 scale);
    scores = mm->add_instruction(migraphx::make_op("mul"), scores, scale);
    auto probs = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), scores);
    auto out   = mm->add_instruction(migraphx::make_op("dot"), probs, vc);
    mm->add_return({out});

    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);

    auto fill = [](const migraphx::shape& s, auto f) {
        std::vector<float> data(s.elements());
        migraphx::shape_for_each(s, [&](const auto& idx) { data[s.index(idx)] = f(idx); });
        return migraphx::literal{s, data}.get_argument();
    };
    auto twenty_four = [](const auto&) { return 24.0f; };
    auto result = p.eval({{"q", fill(q_s, twenty_four)},
                          {"past_k", fill(past_s, twenty_four)},
                          {"k", fill(new_s, twenty_four)},
                          {"past_v", fill(past_s, [](const auto& idx) { return float(idx[2]); })},
                          {"v", fill(new_s, [&](const auto&) { return float(past); })}})
                      .back();

    std::vector<float> out_data;
    result.visit([&](auto x) { out_data.assign(x.begin(), x.end()); });
    const float expected = (kv - 1) / 2.0f;
    EXPECT(std::all_of(
        out_data.begin(), out_data.end(), [&](float x) { return std::abs(x - expected) < 1e-2f; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
