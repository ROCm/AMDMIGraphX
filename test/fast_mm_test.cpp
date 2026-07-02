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
#include <migraphx/fast_mm.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>

#include <algorithm>

#include <test.hpp>

static void run_pass(migraphx::module& m, migraphx::fast_mm fmm = {})
{
    migraphx::run_passes(m, {fmm, migraphx::dead_code_elimination{}});
}

TEST_CASE(fp32_convolution_const_weights_rewritten)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1, {.skip_small_k = 0, .error_threshold = 0.01});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::literal{ws, w_data});

        auto w_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
        auto w_hi_b =
            m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), w_hi_h);
        auto w_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), w_hi_b);
        auto w_lo_f = m2.add_instruction(migraphx::make_op("sub"), w, w_hi_f);
        auto w_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w_lo_f);
        auto w_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), w_hi_h, w_lo_h);

        auto x_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto x_unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x_h);
        auto x_bc   = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                                {{"out_lens", std::vector<std::size_t>{1, 2, 3, 8, 8}}}),
            x_unsq);
        auto x_doubled = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<std::int64_t>{1, 6, 8, 8}}}), x_bc);

        // quant_convolution consumes the fp16 operands and produces the fp32 output directly.
        auto conv = m2.add_instruction(migraphx::make_op("quant_convolution"), x_doubled, w_concat);
        m2.add_return({conv});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_const_weights_three_product)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    // error_threshold = 0 forces every op onto the precision-sensitive path so the
    // three_product flag drives the scheme (here: 3-product).
    run_pass(m1, {.skip_small_k = 0, .three_product = true, .error_threshold = 0});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::literal{ws, w_data});

        // Split the constant weights into hi/lo halves laid out as [hi, lo, hi].
        auto w_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
        auto w_hi_b =
            m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), w_hi_h);
        auto w_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), w_hi_b);
        auto w_lo_f = m2.add_instruction(migraphx::make_op("sub"), w, w_hi_f);
        auto w_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w_lo_f);
        auto w_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), w_hi_h, w_lo_h, w_hi_h);

        // Split the input into hi/lo halves laid out as [hi, hi, lo].
        auto x_hi = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto x_hi_b = m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), x_hi);
        auto x_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x_hi_b);
        auto x_lo_f = m2.add_instruction(migraphx::make_op("sub"), x, x_hi_f);
        auto x_lo   = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x_lo_f);
        auto x_side =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), x_hi, x_hi, x_lo);

        // quant_convolution consumes the fp16 operands and produces the fp32 output directly.
        auto conv = m2.add_instruction(migraphx::make_op("quant_convolution"), x_side, w_concat);
        m2.add_return({conv});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_sensitive_kept_in_fp32)
{
    // A precision-sensitive op (forced via error_threshold = 0) is left in fp32 when the
    // three_product flag is not set, rather than falling back to the 2-product scheme.
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1, {.skip_small_k = 0, .three_product = false, .error_threshold = 0});
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_heuristic_uses_two_product)
{
    // With benign weights the estimated 2-product error is below the threshold, so the
    // op uses 2-product regardless of the three_product flag.
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    // Below error_threshold the op is not precision-sensitive, so both flag settings
    // produce the same 2-product rewrite.
    run_pass(m1, {.skip_small_k = 0, .three_product = true, .error_threshold = 0.01});
    run_pass(m2, {.skip_small_k = 0, .three_product = false, .error_threshold = 0.01});
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_heuristic_uses_output_size)
{
    // Identical weights, but a larger output crosses the threshold while a smaller one
    // stays under it (allclose checks every output element, so more of them raises the
    // estimated worst-case error). The small conv is rewritten; the large one is left in
    // fp32 (three_product not set).
    migraphx::shape ws{migraphx::shape::float_type, {8, 8, 1, 1}};
    std::vector<float> w_data(ws.elements(), 1.0f);
    auto build = [&](std::size_t hw) {
        migraphx::module m;
        auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, hw, hw}});
        auto w    = m.add_literal(migraphx::literal{ws, w_data});
        auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
        m.add_return({conv});
        return m;
    };
    auto has_quant = [](migraphx::module& m) {
        return std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
            return ins.name() == "quant_convolution";
        });
    };
    auto small = build(4);  // ~128 outputs -> est below threshold
    auto large = build(64); // ~32768 outputs -> est above threshold
    run_pass(small, {.skip_small_k = 0, .error_threshold = 5e-3});
    run_pass(large, {.skip_small_k = 0, .error_threshold = 5e-3});
    EXPECT(has_quant(small));
    EXPECT(not has_quant(large));
}

TEST_CASE(fp32_convolution_tiny_unchanged)
{
    // Reduction K = 8 (in_channels * kh * kw) is below the default skip_small_k
    // threshold, so the conv is left untouched.
    migraphx::shape xs{migraphx::shape::float_type, {1, 8, 1, 1}};
    migraphx::shape ws{migraphx::shape::float_type, {11, 8, 1, 1}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_param_weights_unchanged)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(fp16_convolution_unchanged)
{
    migraphx::shape xs{migraphx::shape::half_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::half_type, {4, 3, 3, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(non_convolution_unchanged)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({add});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_dot_const_b_rewritten)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};
    std::vector<float> b_data(bs.elements(), 0.5f);

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_literal(migraphx::literal{bs, b_data});
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1, {.skip_small_k = 0, .error_threshold = 0.01});

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", as);
        auto b = m2.add_literal(migraphx::literal{bs, b_data});

        // Split the constant B along its contraction axis (axis 0).
        auto b_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), b);
        auto b_hi_b =
            m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), b_hi_h);
        auto b_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), b_hi_b);
        auto b_lo_f = m2.add_instruction(migraphx::make_op("sub"), b, b_hi_f);
        auto b_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), b_lo_f);
        auto b_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), b_hi_h, b_lo_h);

        // Duplicate A along its contraction axis (axis 1).
        auto a_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), a);
        auto a_unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a_h);
        auto a_bc   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", std::vector<std::size_t>{2, 2, 8}}}),
            a_unsq);
        auto a_doubled = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<std::int64_t>{2, 16}}}), a_bc);

        // quant_dot consumes the fp16 operands and produces the fp32 output directly.
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), a_doubled, b_concat);
        m2.add_return({dot});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_dot_const_a_rewritten)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};
    std::vector<float> a_data(as.elements(), 0.5f);

    migraphx::module m1;
    {
        auto a   = m1.add_literal(migraphx::literal{as, a_data});
        auto b   = m1.add_parameter("b", bs);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1, {.skip_small_k = 0, .error_threshold = 0.01});

    migraphx::module m2;
    {
        auto a = m2.add_literal(migraphx::literal{as, a_data});
        auto b = m2.add_parameter("b", bs);

        // Split the constant A along its contraction axis (axis 1).
        auto a_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), a);
        auto a_hi_b =
            m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), a_hi_h);
        auto a_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), a_hi_b);
        auto a_lo_f = m2.add_instruction(migraphx::make_op("sub"), a, a_hi_f);
        auto a_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), a_lo_f);
        auto a_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), a_hi_h, a_lo_h);

        // Duplicate B along its contraction axis (axis 0).
        auto b_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), b);
        auto b_unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b_h);
        auto b_bc   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", std::vector<std::size_t>{2, 8, 4}}}),
            b_unsq);
        auto b_doubled = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<std::int64_t>{16, 4}}}), b_bc);

        // quant_dot consumes the fp16 operands and produces the fp32 output directly.
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), a_concat, b_doubled);
        m2.add_return({dot});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_dot_const_b_three_product)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};
    std::vector<float> b_data(bs.elements(), 0.5f);

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_literal(migraphx::literal{bs, b_data});
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    // error_threshold = 0 forces every op onto the precision-sensitive path so the
    // three_product flag drives the scheme (here: 3-product).
    run_pass(m1, {.skip_small_k = 0, .three_product = true, .error_threshold = 0});

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", as);
        auto b = m2.add_literal(migraphx::literal{bs, b_data});

        // Split the constant B along its contraction axis (axis 0) as [hi, lo, hi].
        auto b_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), b);
        auto b_hi_b =
            m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), b_hi_h);
        auto b_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), b_hi_b);
        auto b_lo_f = m2.add_instruction(migraphx::make_op("sub"), b, b_hi_f);
        auto b_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), b_lo_f);
        auto b_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), b_hi_h, b_lo_h, b_hi_h);

        // Split A along its contraction axis (axis 1) as [hi, hi, lo].
        auto a_hi = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), a);
        auto a_hi_b = m2.add_instruction(migraphx::make_op("barrier", {{"tag", "fast_mm"}}), a_hi);
        auto a_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), a_hi_b);
        auto a_lo_f = m2.add_instruction(migraphx::make_op("sub"), a, a_hi_f);
        auto a_lo   = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), a_lo_f);
        auto a_side =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), a_hi, a_hi, a_lo);

        // quant_dot consumes the fp16 operands and produces the fp32 output directly.
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), a_side, b_concat);
        m2.add_return({dot});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_dot_param_operands_unchanged)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_parameter("b", bs);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    auto m2 = m1;
    run_pass(m1, {.skip_small_k = 0, .error_threshold = 0.01});
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_dot_tiny_k_unchanged)
{
    // K = 8 is below the default skip_small_k threshold, so leave it untouched.
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};
    std::vector<float> b_data(bs.elements(), 0.5f);

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_literal(migraphx::literal{bs, b_data});
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Build dot(activation(param), const_weights) where `add_activation` maps the parameter to
// the dot's left operand, run fast_mm, and report whether the dot was rewritten to
// quant_dot. With input_bound = 4 a raw parameter's estimated error is over the threshold
// (skip); an activation whose magnitude is provably smaller narrows the bound and drops the
// estimate below the threshold (2-product rewrite). Identical weights and threshold across
// activations, so the only variable is the narrowed bound.
template <class F>
static bool narrowed_dot_is_rewritten(F add_activation)
{
    migraphx::shape as{migraphx::shape::float_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 4}};
    std::vector<float> b_data(bs.elements(), 1.0f);

    migraphx::module m;
    auto a   = m.add_parameter("a", as);
    auto act = add_activation(m, a);
    auto b   = m.add_literal(migraphx::literal{bs, b_data});
    auto dot = m.add_instruction(migraphx::make_op("dot"), act, b);
    m.add_return({dot});
    run_pass(m, {.skip_small_k = 0, .input_bound = 4, .error_threshold = 5e-3});
    return std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return ins.name() == "quant_dot";
    });
}

TEST_CASE(fp32_dot_activation_bound_param_not_narrowed)
{
    // A raw parameter has no static magnitude bound, so the pass falls back to input_bound
    // (4); the estimated error is over the threshold and the dot is left in fp32.
    EXPECT(not narrowed_dot_is_rewritten(
        [](migraphx::module&, migraphx::instruction_ref a) { return a; }));
}

TEST_CASE(fp32_dot_activation_bound_narrowed_by_sigmoid)
{
    // sigmoid output is provably in (0, 1), narrowing the bound to 1 so the estimate drops
    // below the threshold and the dot is rewritten with 2-product.
    EXPECT(narrowed_dot_is_rewritten([](migraphx::module& m, migraphx::instruction_ref a) {
        return m.add_instruction(migraphx::make_op("sigmoid"), a);
    }));
}

TEST_CASE(fp32_dot_activation_bound_narrowed_by_softmax)
{
    // softmax normalizes to a probability distribution in [0, 1], so the bound narrows to 1
    // regardless of the input magnitude and the dot is rewritten.
    EXPECT(narrowed_dot_is_rewritten([](migraphx::module& m, migraphx::instruction_ref a) {
        return m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), a);
    }));
}

TEST_CASE(fp32_dot_activation_bound_narrowed_by_clip)
{
    // Min-max normalization: clip bounds the activation to [-0.5, 0.5] independent of the
    // input, narrowing the magnitude bound to 0.5 and the dot is rewritten.
    EXPECT(narrowed_dot_is_rewritten([](migraphx::module& m, migraphx::instruction_ref a) {
        migraphx::shape as{migraphx::shape::float_type, {2, 8}};
        auto lo = m.add_literal(migraphx::literal{as, std::vector<float>(as.elements(), -0.5f)});
        auto hi = m.add_literal(migraphx::literal{as, std::vector<float>(as.elements(), 0.5f)});
        return m.add_instruction(migraphx::make_op("clip"), a, lo, hi);
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
