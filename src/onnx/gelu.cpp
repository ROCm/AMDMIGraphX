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
#include <migraphx/onnx/gelu.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

instruction_ref add_gelu(onnx_parser::node_info info, instruction_ref x)
{
    // use the erf version for both tanh and erf version
    // rewrite_gelu will rewrite to use tanh as needed
    auto x_lens = x->get_shape().lens();
    auto x_type = x->get_shape().type();
    auto half   = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {0.5f}});
    auto one    = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {1.0f}});
    auto sqrt2 =
        info.add_literal(migraphx::literal{migraphx::shape{x_type}, {static_cast<float>(M_SQRT2)}});
    auto half_mbcast =
        info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), half);
    auto mul_half = info.add_instruction(migraphx::make_op("mul"), x, half_mbcast);
    auto sqrt2_mbcast =
        info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), sqrt2);
    auto div = info.add_instruction(migraphx::make_op("div"), x, sqrt2_mbcast);
    auto erf = info.add_instruction(migraphx::make_op("erf"), div);
    auto one_mbcast =
        info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), one);
    auto add_one = info.add_instruction(migraphx::make_op("add"), erf, one_mbcast);
    return info.add_instruction(migraphx::make_op("mul"), mul_half, add_one);
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
