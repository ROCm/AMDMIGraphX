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
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/shape_for_each.hpp>
#include <test.hpp>
#include <cstdint>
#include <numeric>
#include <vector>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_mlir{}});
}

// Square mask literal with `above` strictly above the diagonal and `other`
// everywhere else
static migraphx::literal make_triangle_literal(const migraphx::shape& s, float above, float other)
{
    std::vector<float> data(s.elements());
    migraphx::shape_for_each(
        s, [&](const auto& idx, std::size_t i) { data[i] = idx[1] > idx[0] ? above : other; });
    return migraphx::literal{s, data};
}

static migraphx::literal make_iota_literal(std::size_t n)
{
    std::vector<std::int32_t> iota_vals(n);
    std::iota(iota_vals.begin(), iota_vals.end(), 0);
    return migraphx::literal{{migraphx::shape::int32_type, {n}}, iota_vals};
}

TEST_CASE(is_causal_mask_literal_detection)
{
    migraphx::shape bs{migraphx::shape::bool_type, {4, 4}};
    migraphx::shape hs{migraphx::shape::half_type, {4, 4}};
    // boolean zero-mask: ones strictly above the diagonal
    EXPECT(migraphx::gpu::is_causal_mask_literal(make_triangle_literal(bs, 1.0f, 0.0f)));
    // multiplicative keep-mask: ones on and below the diagonal
    EXPECT(migraphx::gpu::is_causal_mask_literal(make_triangle_literal(hs, 0.0f, 1.0f)));
    // wrong polarity for the type
    EXPECT(not migraphx::gpu::is_causal_mask_literal(make_triangle_literal(bs, 0.0f, 1.0f)));
    EXPECT(not migraphx::gpu::is_causal_mask_literal(make_triangle_literal(hs, 1.0f, 0.0f)));
    // not a 0/1 mask
    EXPECT(not migraphx::gpu::is_causal_mask_literal(make_triangle_literal(bs, 1.0f, 1.0f)));
    EXPECT(not migraphx::gpu::is_causal_mask_literal(make_triangle_literal(hs, 0.0f, 2.0f)));
    // not square, not 2d
    EXPECT(not migraphx::gpu::is_causal_mask_literal(
        make_triangle_literal({migraphx::shape::half_type, {4, 2}}, 0.0f, 1.0f)));
    EXPECT(not migraphx::gpu::is_causal_mask_literal(make_iota_literal(4)));
}

TEST_CASE(causal_mask_literal_where)
// An inlined boolean triangle mask feeding a where predicate becomes a
// compare of index iotas sunk to the mask's broadcast shape
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto mask =
            m1.add_literal(make_triangle_literal({migraphx::shape::bool_type, {4, 4}}, 1.0f, 0.0f));
        auto mask_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), mask);
        auto zero   = m1.add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0}});
        auto zero_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), zero);
        auto w = m1.add_instruction(migraphx::make_op("where"), mask_b, zero_b, x);
        m1.add_return({w});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto zero = m2.add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0}});
        auto iota = m2.add_literal(make_iota_literal(4));
        auto rows = m2.insert_instruction(x, migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto cols_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), iota);
        auto rows_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), rows);
        auto gt   = m2.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto zero_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), zero);
        auto w = m2.add_instruction(migraphx::make_op("where"), pred, zero_b, x);
        m2.add_return({w});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(causal_mask_literal_mul)
// An inlined multiplicative triangle mask feeding a mul becomes the
// where-zero-above select over a compare of index iotas
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {2, 4, 4}});
        auto mask =
            m1.add_literal(make_triangle_literal({migraphx::shape::half_type, {4, 4}}, 0.0f, 1.0f));
        auto mask_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), mask);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, mask_b);
        m1.add_return({mul});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", {migraphx::shape::half_type, {2, 4, 4}});
        auto iota = m2.add_literal(make_iota_literal(4));
        auto zero = m2.add_literal(migraphx::literal{{migraphx::shape::half_type, {1}}, {0.0}});
        auto rows = m2.insert_instruction(x, migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto cols_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), iota);
        auto rows_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), rows);
        auto gt   = m2.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto zero_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), zero);
        auto w = m2.add_instruction(migraphx::make_op("where"), pred, zero_b, x);
        m2.add_return({w});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(masked_mul_fold)
// A mul by a broadcast where(pred, 0, 1) mask folds into where(pred, 0, x)
// with the predicate rebuilt at the mul's shape
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::half_type, {2, 4, 4}});
        auto iota = m1.add_literal(make_iota_literal(4));
        auto rows = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto rows_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), rows);
        auto cols_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), iota);
        auto gt   = m1.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto zero = m1.add_literal(migraphx::literal{{migraphx::shape::half_type, {1}}, {0.0}});
        auto one  = m1.add_literal(migraphx::literal{{migraphx::shape::half_type, {1}}, {1.0}});
        auto zero_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), zero);
        auto one_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), one);
        auto mask   = m1.add_instruction(migraphx::make_op("where"), pred, zero_b, one_b);
        auto mask_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), mask);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, mask_b);
        m1.add_return({mul});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::half_type, {2, 4, 4}});
        auto iota   = m2.add_literal(make_iota_literal(4));
        auto zero   = m2.add_literal(migraphx::literal{{migraphx::shape::half_type, {1}}, {0.0}});
        auto rows   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto cols_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), iota);
        auto rows_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), rows);
        auto gt   = m2.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto zero_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), zero);
        auto w = m2.add_instruction(migraphx::make_op("where"), pred, zero_b, x);
        m2.add_return({w});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_of_compare_sink)
// A broadcast of a compare of broadcast iotas is sunk below the compare so
// the consumer sees the compare directly
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto y    = m1.add_parameter("y", {migraphx::shape::float_type, {2, 4, 4}});
        auto iota = m1.add_literal(make_iota_literal(4));
        auto rows = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto rows_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), rows);
        auto cols_b =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), iota);
        auto gt   = m1.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto pred_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), pred);
        auto w = m1.add_instruction(migraphx::make_op("where"), pred_b, x, y);
        m1.add_return({w});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto y      = m2.add_parameter("y", {migraphx::shape::float_type, {2, 4, 4}});
        auto iota   = m2.add_literal(make_iota_literal(4));
        auto rows   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), iota);
        auto cols_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), iota);
        auto rows_b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), rows);
        auto gt   = m2.add_instruction(migraphx::make_op("greater"), cols_b, rows_b);
        auto pred = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto w = m2.add_instruction(migraphx::make_op("where"), pred, x, y);
        m2.add_return({w});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(non_triangle_literal_unchanged)
// A dense bool literal that is not a causal mask is left alone
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto mask =
            m1.add_literal(make_triangle_literal({migraphx::shape::bool_type, {4, 4}}, 1.0f, 1.0f));
        auto mask_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), mask);
        auto zero   = m1.add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0}});
        auto zero_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), zero);
        auto w = m1.add_instruction(migraphx::make_op("where"), mask_b, zero_b, x);
        m1.add_return({w});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(non_unit_mul_mask_unchanged)
// A mul by a triangle of other values than 0/1 is not a causal mask
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {2, 4, 4}});
        auto mask =
            m1.add_literal(make_triangle_literal({migraphx::shape::half_type, {4, 4}}, 0.0f, 2.0f));
        auto mask_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), mask);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, mask_b);
        m1.add_return({mul});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(compare_without_broadcast_inputs_unchanged)
// The sink only applies when the compare's operands are broadcasts that can
// be rebuilt at the outer shape
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4, 4}});
        auto y    = m1.add_parameter("y", {migraphx::shape::float_type, {2, 4, 4}});
        auto a    = m1.add_parameter("a", {migraphx::shape::int32_type, {4, 4}});
        auto b    = m1.add_parameter("b", {migraphx::shape::int32_type, {4, 4}});
        auto gt   = m1.add_instruction(migraphx::make_op("greater"), a, b);
        auto pred = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), gt);
        auto pred_b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4}}}), pred);
        auto w = m1.add_instruction(migraphx::make_op("where"), pred_b, x, y);
        m1.add_return({w});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
