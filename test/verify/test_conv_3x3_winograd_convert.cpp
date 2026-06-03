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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// The F(2,3) winograd input is fp16 (the matcher only fires on half input),
// but the fused pointwise can convert the result to a *different* final
// output type. The packed-store writeback must use the actual output type,
// not assume fp16 — otherwise the store width / packing is wrong for a
// non-fp16 output. This fuses conv -> bias add -> leaky_relu -> convert(Out)
// so the winograd kernel's output buffer type is Out, exercising the
// writeback for each Out. (Run with MIGRAPHX_ENABLE_WINOGRAD=1 to exercise
// the winograd path; otherwise it validates the default lowering.)
//
// Covered: fp16 (the convert is a no-op control) and fp32 (a wider final
// type, so the writeback must NOT assume a 2-byte fp16 store). bf16 output is
// intentionally not covered here: the default (non-winograd) lowering can't
// even compile an fp16->bf16 conv fusion (MLIR rejects `arith.truncf f16 ->
// bf16`), so it cannot run in the default CI config — the winograd bf16 path
// itself is correct and was validated separately.
//
// Tolerance: winograd computes the transforms in fp16, so its fp32 output
// carries fp16-magnitude error. Like the other reduced-precision verify
// tests (mxfp4, quantizelinear), the fp32 case overrides the tolerance to an
// fp16-appropriate bound; the default fp32 lowering passes it comfortably.
template <migraphx::shape::type_t Out>
struct test_conv_3x3_winograd_convert : verify_program<test_conv_3x3_winograd_convert<Out>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // 14x14 -> 7x7 tiles: even width hits the packed two-output store, and
        // the edges still produce border tiles.
        auto x = mm->add_parameter("x", {migraphx::shape::half_type, {1, 64, 14, 14}});
        // Winograd matcher requires can_eval() on weights -> add as literals.
        auto w = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {64, 64, 3, 3}}, 1));
        auto bias =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {64}}, 2));
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto bias_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto add = mm->add_instruction(migraphx::make_op("add"), conv, bias_b);
        auto act = mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 0.2}}), add);
        mm->add_instruction(migraphx::make_op("convert", {{"target_type", Out}}), act);
        return p;
    }
    std::string section() const { return "conv"; }
    // fp16 output uses the standard tolerance; fp32 output must allow for
    // winograd's fp16-magnitude transform error (see header comment).
    std::size_t get_tolerance() const { return Out == migraphx::shape::half_type ? 80 : 80000; }
};

// fp16 output: the convert is a no-op; exercises the 2-byte packed store.
template struct test_conv_3x3_winograd_convert<migraphx::shape::half_type>;
// fp32 output: a wider final type — the writeback must use the real output
// type (4-byte store), not assume a packed fp16 store.
template struct test_conv_3x3_winograd_convert<migraphx::shape::float_type>;
