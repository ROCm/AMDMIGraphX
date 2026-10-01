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

#include <vector>
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_mlir{}, migraphx::dead_code_elimination{}});
}

// A non-standard-strided literal (as folded from a transposed constant, which mlir rejects) is
// rewritten to a standard shape, preserving the logical values.
TEST_CASE(nonstandard_literal_normalized)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module m1;
    {
        migraphx::shape s{f, {2, 2}, {1, 2}};
        auto lit = m1.add_literal(migraphx::literal{s, {1, 3, 2, 4}});
        m1.add_return({lit});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto lit = m2.add_literal(migraphx::literal{migraphx::shape{f, {2, 2}}, {1, 3, 2, 4}});
        m2.add_return({lit});
    }

    EXPECT(m1.sort() == m2.sort());
}

// An already-standard literal is left untouched.
TEST_CASE(standard_literal_unchanged)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module m1;
    {
        auto lit = m1.add_literal(migraphx::literal{migraphx::shape{f, {2, 2}}, {1, 2, 3, 4}});
        m1.add_return({lit});
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// The kv-cache mask sequence length is broadcast over the leading batch and
// heads dimensions in a separate step so rocMLIR can bind a {batch, heads}
// tensor that matches the attention batch.
TEST_CASE(kv_cache_mask_seq_len)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;
    migraphx::shape ss{i, {1, 1}};
    migraphx::shape scores_s{f, {1, 3, 1, 4}};

    migraphx::module m1;
    {
        auto scores  = m1.add_parameter("scores", scores_s);
        auto seq_len = m1.add_parameter("seq_len", ss);
        auto iota    = m1.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 4}}}), iota);
        auto rsl = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1}}}), seq_len);
        auto bsl = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 1, 1, 4}}}), rsl);
        auto gt  = m1.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto bcond = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), cvt);
        auto ninf = m1.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m1.add_instruction(migraphx::make_op("where"), bcond, binf, scores);
        m1.add_return({w});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto scores  = m2.add_parameter("scores", scores_s);
        auto seq_len = m2.add_parameter("seq_len", ss);
        auto iota    = m2.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), iota);
        auto flat = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1}}}), seq_len);
        auto lead =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 3}}}), flat);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), lead);
        auto bsl  = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), unsq);
        auto gt  = m2.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto bcond = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), cvt);
        auto ninf = m2.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m2.add_instruction(migraphx::make_op("where"), bcond, binf, scores);
        m2.add_return({w});
    }

    EXPECT(m1.sort() == m2.sort());
}

// Running the pass a second time makes no further changes
TEST_CASE(kv_cache_mask_seq_len_idempotent)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;
    migraphx::shape ss{i, {1, 1}};
    migraphx::shape scores_s{f, {1, 3, 1, 4}};

    migraphx::module m1;
    {
        auto scores  = m1.add_parameter("scores", scores_s);
        auto seq_len = m1.add_parameter("seq_len", ss);
        auto iota    = m1.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 4}}}), iota);
        auto rsl = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1}}}), seq_len);
        auto bsl = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 1, 1, 4}}}), rsl);
        auto gt  = m1.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto bcond = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), cvt);
        auto ninf = m1.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m1.add_instruction(migraphx::make_op("where"), bcond, binf, scores);
        m1.add_return({w});
    }
    run_pass(m1);
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// A batched sequence length holds one value per batch; it is broadcast over
// the heads only, so each batch keeps its own length after rocMLIR folds the
// heads into the attention batch.
TEST_CASE(kv_cache_mask_seq_len_batched)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;
    migraphx::shape ss{i, {2, 1}};
    migraphx::shape scores_s{f, {2, 3, 1, 4}};

    migraphx::module m1;
    {
        auto scores  = m1.add_parameter("scores", scores_s);
        auto seq_len = m1.add_parameter("seq_len", ss);
        auto iota    = m1.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), iota);
        auto rsl =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 1, 1}}}), seq_len);
        auto bsl = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), rsl);
        auto gt  = m1.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto ninf = m1.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m1.add_instruction(migraphx::make_op("where"), cvt, binf, scores);
        m1.add_return({w});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto scores  = m2.add_parameter("scores", scores_s);
        auto seq_len = m2.add_parameter("seq_len", ss);
        auto iota    = m2.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), iota);
        auto flat = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1}}}), seq_len);
        auto lead =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), flat);
        auto unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), lead);
        auto bsl  = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), unsq);
        auto gt  = m2.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto ninf = m2.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m2.add_instruction(migraphx::make_op("where"), cvt, binf, scores);
        m2.add_return({w});
    }

    EXPECT(m1.sort() == m2.sort());
}

// A mask with no heads to broadcast over is left untouched
TEST_CASE(kv_cache_mask_seq_len_no_heads)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;
    migraphx::shape ss{i, {1, 1}};
    migraphx::shape scores_s{f, {1, 1, 1, 4}};

    migraphx::module m1;
    {
        auto scores  = m1.add_parameter("scores", scores_s);
        auto seq_len = m1.add_parameter("seq_len", ss);
        auto iota    = m1.add_literal(migraphx::literal{migraphx::shape{i, {4}}, {0, 1, 2, 3}});
        auto biota   = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 4}}}), iota);
        auto rsl = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1}}}), seq_len);
        auto bsl = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {1, 1, 1, 4}}}), rsl);
        auto gt  = m1.add_instruction(migraphx::make_op("greater"), biota, bsl);
        auto cvt = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto ninf = m1.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
        auto binf = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores_s.lens()}}), ninf);
        auto w = m1.add_instruction(migraphx::make_op("where"), cvt, binf, scores);
        m1.add_return({w});
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// Scaled decode attention scores dot(q, k) under a kv-cache mask, with the
// sequence length already broadcast the way find_kv_cache_mask_seq_len leaves it
static void add_kv_cache_scores(migraphx::module& m, migraphx::instruction_ref q)
{
    const auto f      = migraphx::shape::float_type;
    const auto i      = migraphx::shape::int32_type;
    const auto& lens  = q->get_shape().lens();
    const auto batch  = lens[0];
    const auto heads  = lens[1];
    const auto seq    = lens[2];
    const auto dim    = lens[3];
    const auto kv_len = std::size_t{8};
    std::vector<std::size_t> scores_lens{batch, heads, seq, kv_len};

    auto k       = m.add_parameter("k", {f, {batch, heads, dim, kv_len}});
    auto seq_len = m.add_parameter("seq_len", {i, {batch, 1}});
    auto gemm    = m.add_instruction(migraphx::make_op("dot"), q, k);
    auto scale   = m.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {0.125f}});
    auto bscale =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}), scale);
    auto scaled = m.add_instruction(migraphx::make_op("mul"), gemm, bscale);
    auto iota =
        m.add_literal(migraphx::literal{migraphx::shape{i, {kv_len}}, {0, 1, 2, 3, 4, 5, 6, 7}});
    auto biota =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}), iota);
    auto flat = m.add_instruction(migraphx::make_op("reshape", {{"dims", {batch, 1}}}), seq_len);
    auto lead = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {batch, heads}}}), flat);
    auto unsq = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 3}}}), lead);
    auto bsl =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}), unsq);
    auto gt  = m.add_instruction(migraphx::make_op("greater"), biota, bsl);
    auto cvt = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
    auto ninf = m.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {-1e9f}});
    auto binf =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", scores_lens}}), ninf);
    auto w = m.add_instruction(migraphx::make_op("where"), cvt, binf, scaled);
    m.add_return({w});
}

// A plain decode-shaped Q parameter gets a transpose producer that swaps its
// unit sequence dimension, so rocMLIR keeps the collapse it reads the
// per-head sequence length broadcast from.
TEST_CASE(kv_cache_plain_q)
{
    migraphx::shape q_s{migraphx::shape::float_type, {1, 4, 1, 4}};

    migraphx::module m1;
    {
        auto q = m1.add_parameter("q", q_s);
        add_kv_cache_scores(m1, q);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto q   = m2.add_parameter("q", q_s);
        auto rsp = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 4}}}), q);
        auto tsp = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsp);
        add_kv_cache_scores(m2, tsp);
    }

    EXPECT(m1.sort() == m2.sort());
}

// Running the pass a second time makes no further changes
TEST_CASE(kv_cache_plain_q_idempotent)
{
    migraphx::shape q_s{migraphx::shape::float_type, {2, 4, 1, 4}};

    migraphx::module m1;
    {
        auto q = m1.add_parameter("q", q_s);
        add_kv_cache_scores(m1, q);
    }
    run_pass(m1);
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// A Q that already has a producer is left untouched
TEST_CASE(kv_cache_q_with_producer)
{
    migraphx::shape q_s{migraphx::shape::float_type, {1, 1, 4, 4}};

    migraphx::module m1;
    {
        auto q_in = m1.add_parameter("q", q_s);
        auto q = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                    q_in);
        add_kv_cache_scores(m1, q);
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// A prefill-shaped Q has no unit dimension to swap for free, so it is left alone
TEST_CASE(kv_cache_plain_q_prefill)
{
    migraphx::shape q_s{migraphx::shape::float_type, {1, 4, 3, 4}};

    migraphx::module m1;
    {
        auto q = m1.add_parameter("q", q_s);
        add_kv_cache_scores(m1, q);
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

// With a single head the sequence length already matches the attention batch
TEST_CASE(kv_cache_plain_q_single_head)
{
    migraphx::shape q_s{migraphx::shape::float_type, {2, 1, 1, 4}};

    migraphx::module m1;
    {
        auto q = m1.add_parameter("q", q_s);
        add_kv_cache_scores(m1, q);
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
