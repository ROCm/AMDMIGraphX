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

// Standalone reproducer for why remove_qdq_pairs (src/simplify_qdq.cpp) must
// NOT collapse a quantizelinear -> dequantizelinear round-trip when it
// represents a "residual-accumulate-then-requantize" idiom: a genuine
// saturating clamp performed before a residual add in a fixed-function
// quantized-conv kernel. See is_residual_requant_accumulator in
// src/simplify_qdq.cpp for the guard this file exercises.
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <quantize_helpers.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/target.hpp>

static void run_pass(migraphx::module& m)
{
    run_passes(m,
               {migraphx::simplify_qdq{.remove_qdq_only = false, .use_mx_quant = true},
                migraphx::dead_code_elimination{}});
}

// Reproduces the "residual-accumulate-then-requantize" idiom emitted by a
// fixed-function two-stage quantized-conv-plus-residual kernel:
//
//   t_main -> QL(mid) -> DQ(mid) -> add <- DQ(residual) <- QL(residual) <- t_res
//                                     |
//                                     v
//                                  QL(final)
//
// remove_qdq_pairs must NOT cancel the QL(mid)/DQ(mid) round-trip here: it
// represents a real saturating clamp that happens before the residual add in
// the fused kernel. Without is_residual_requant_accumulator, this round-trip
// would be collapsed just like the generic `remove_qdq` case, silently
// discarding that clamp and leaving the later fusion pass with no way to
// recover the true intermediate scale/zero_point.
TEST_CASE(residual_requant_accumulator_preserved)
{
    migraphx::shape sh{migraphx::shape::float_type, {100, 100}};

    migraphx::module m1;
    {
        auto t_main    = m1.add_parameter("t_main", sh);
        auto t_res     = m1.add_parameter("t_res", sh);
        auto scale_mid = m1.add_literal(0.5f);
        // Non-zero zero-points -- if these were 0, simplify_qdq's unrelated
        // remove_zero_point cleanup would drop the (redundant, always-zero)
        // zero-point argument from the dequantizelinear ops, which changes
        // m1's instruction shape without touching the round-trip itself and
        // would make the m1 == m2 comparison below fail for an unrelated
        // reason.
        auto zero_mid  = m1.add_literal(std::int8_t{1});
        auto scale_res = m1.add_literal(0.25f);
        auto zero_res  = m1.add_literal(std::int8_t{1});
        auto scale_out = m1.add_literal(0.75f);
        auto zero_out  = m1.add_literal(std::int8_t{1});

        auto q_mid = add_quantize_op(m1, "quantizelinear", t_main, scale_mid, zero_mid);
        auto d_mid = add_quantize_op(m1, "dequantizelinear", q_mid, scale_mid, zero_mid);
        auto q_res = add_quantize_op(m1, "quantizelinear", t_res, scale_res, zero_res);
        auto d_res = add_quantize_op(m1, "dequantizelinear", q_res, scale_res, zero_res);
        auto add   = m1.add_instruction(migraphx::make_op("add"), d_mid, d_res);
        auto q_out = add_quantize_op(m1, "quantizelinear", add, scale_out, zero_out);
        m1.add_return({q_out});
    }

    migraphx::module m2 = m1;

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Negative control: the "residual" side here is a plain float parameter, not
// a dequantizelinear, so this is just an ordinary add rather than a genuine
// residual-accumulate idiom. is_residual_requant_accumulator must NOT fire,
// so QL(mid)/DQ(mid) is safely collapsed by the normal remove_qdq_pairs
// behavior -- confirming the guard above is narrowly scoped and doesn't
// suppress the optimization in the common case.
TEST_CASE(residual_requant_accumulator_not_triggered_without_residual_dq)
{
    migraphx::shape sh{migraphx::shape::float_type, {100, 100}};

    migraphx::module m1;
    {
        auto t_main    = m1.add_parameter("t_main", sh);
        auto t_res     = m1.add_parameter("t_res", sh);
        auto scale_mid = m1.add_literal(0.5f);
        auto zero_mid  = m1.add_literal(std::int8_t{0});
        auto scale_out = m1.add_literal(0.75f);
        auto zero_out  = m1.add_literal(std::int8_t{0});

        auto q_mid = add_quantize_op(m1, "quantizelinear", t_main, scale_mid, zero_mid);
        auto d_mid = add_quantize_op(m1, "dequantizelinear", q_mid, scale_mid, zero_mid);
        auto add   = m1.add_instruction(migraphx::make_op("add"), d_mid, t_res);
        auto q_out = add_quantize_op(m1, "quantizelinear", add, scale_out, zero_out);
        m1.add_return({q_out});
    }

    migraphx::module m2;
    {
        auto t_main    = m2.add_parameter("t_main", sh);
        auto t_res     = m2.add_parameter("t_res", sh);
        auto scale_out = m2.add_literal(0.75f);
        auto zero_out  = m2.add_literal(std::int8_t{0});

        auto add   = m2.add_instruction(migraphx::make_op("add"), t_main, t_res);
        auto q_out = add_quantize_op(m2, "quantizelinear", add, scale_out, zero_out);
        m2.add_return({q_out});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Numeric demonstration (not just structural): the QL(mid)/DQ(mid) round-trip
// preserved by is_residual_requant_accumulator performs a real round+clamp
// operation, not just scale/zero_point bookkeeping -- it mirrors the real
// fixed-function kernel, which clamps the rescaled conv accumulator to the
// intermediate quantized range *before* adding the residual (see
// src/targets/gpu/jit/qdq-used.cpp's RESIDUAL_ADD epilogue). Skipping the
// round-trip (as remove_qdq_pairs would without the guard) changes the
// computed output whenever the pre-quantize value would have saturated.
TEST_CASE(residual_requant_accumulator_clamp_changes_output)
{
    migraphx::shape sh{migraphx::shape::float_type, {4}};
    migraphx::shape sh_res{migraphx::shape::int8_type, {4}};

    // t_main = 300 saturates int8 (scale=1, zp=0) to 127 -- residual = 10,
    // scale_out = 4 keeps the final requantize itself away from saturating,
    // so the two paths land on distinct, easily distinguishable int8 codes.
    std::vector<float> main_vals(4, 300.0f);
    std::vector<std::int8_t> res_vals(4, 10);

    // p1: real graph, with the QL(mid)/DQ(mid) round-trip run through the
    // actual remove_qdq_pairs pass -- proves it survives (guard fires) AND
    // gives the correct, saturated numeric result.
    migraphx::program p1;
    {
        auto* m1       = p1.get_main_module();
        auto t_main    = m1->add_parameter("t_main", sh);
        auto t_res     = m1->add_parameter("t_res", sh_res);
        auto scale_mid = m1->add_literal(1.0f);
        auto zero_mid  = m1->add_literal(std::int8_t{0});
        auto scale_res = m1->add_literal(1.0f);
        auto zero_res  = m1->add_literal(std::int8_t{0});
        auto scale_out = m1->add_literal(4.0f);
        auto zero_out  = m1->add_literal(std::int8_t{0});

        auto q_mid = add_quantize_op(*m1, "quantizelinear", t_main, scale_mid, zero_mid);
        auto d_mid = add_quantize_op(*m1, "dequantizelinear", q_mid, scale_mid, zero_mid);
        auto d_res = add_quantize_op(*m1, "dequantizelinear", t_res, scale_res, zero_res);
        auto add   = m1->add_instruction(migraphx::make_op("add"), d_mid, d_res);
        auto q_out = add_quantize_op(*m1, "quantizelinear", add, scale_out, zero_out);
        m1->add_return({q_out});

        run_pass(*m1);

        // Confirm the fix actually kept the round-trip -- if this ever
        // regresses, the two programs below would trivially match again for
        // the wrong reason (both would collapse to the unclamped path).
        auto match_qdq = migraphx::match::name("dequantizelinear")(
            migraphx::match::arg(0)(migraphx::match::name("quantizelinear")));
        EXPECT(migraphx::match::find_match(*m1, match_qdq).result != m1->end());
    }

    // p2: exactly what remove_qdq_pairs::apply would produce if the guard
    // did NOT fire -- the mid round-trip bypassed, t_main feeding the add
    // directly in full (unclamped) float precision.
    migraphx::program p2;
    {
        auto* m2       = p2.get_main_module();
        auto t_main    = m2->add_parameter("t_main", sh);
        auto t_res     = m2->add_parameter("t_res", sh_res);
        auto scale_res = m2->add_literal(1.0f);
        auto zero_res  = m2->add_literal(std::int8_t{0});
        auto scale_out = m2->add_literal(4.0f);
        auto zero_out  = m2->add_literal(std::int8_t{0});

        auto d_res = add_quantize_op(*m2, "dequantizelinear", t_res, scale_res, zero_res);
        auto add   = m2->add_instruction(migraphx::make_op("add"), t_main, d_res);
        auto q_out = add_quantize_op(*m2, "quantizelinear", add, scale_out, zero_out);
        m2->add_return({q_out});
    }

    auto t_main_arg = migraphx::argument(sh, main_vals.data());
    auto t_res_arg  = migraphx::argument(sh_res, res_vals.data());

    p1.compile(migraphx::target(migraphx::make_target("ref")));
    p2.compile(migraphx::target(migraphx::make_target("ref")));

    auto result1 = p1.eval({{"t_main", t_main_arg}, {"t_res", t_res_arg}}).back();
    auto result2 = p2.eval({{"t_main", t_main_arg}, {"t_res", t_res_arg}}).back();

    std::vector<int32_t> rv1(4);
    std::vector<int32_t> rv2(4);
    result1.visit([&](auto output) { rv1.assign(output.begin(), output.end()); });
    result2.visit([&](auto output) { rv2.assign(output.begin(), output.end()); });

    // Clamped path: 300 saturates to int8 max (127) before the residual add,
    // giving (127 + 10) / 4 = 34. Bypassed path: unclamped 300 + 10 = 310,
    // giving 310 / 4 ~= 78. These differ by ~44 -- far beyond quantization
    // rounding noise -- proving the round-trip is a real numeric operation,
    // not just a carrier for scale/zero_point.
    EXPECT(rv1 == std::vector<int32_t>(4, 34));
    EXPECT(rv2 == std::vector<int32_t>(4, 78));
    EXPECT(std::abs(rv1.front() - rv2.front()) > 10);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
