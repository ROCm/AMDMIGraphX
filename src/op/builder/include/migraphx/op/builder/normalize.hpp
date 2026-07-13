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

#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_NORMALIZE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_NORMALIZE_HPP

#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// (x - mean) * rsqrt(var + epsilon) reduced over `axes`, with a biased variance.
inline instruction_ref normalize(module& m,
                                 instruction_ref ins,
                                 instruction_ref x,
                                 const std::vector<int64_t>& axes,
                                 float epsilon)
{
    auto x_type   = x->get_shape().type();
    auto mean     = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), x);
    auto x_sub    = insert_common_op(m, ins, "sub", x, mean);
    auto sqdiff   = insert_common_op(m, ins, "sqdiff", x, mean);
    auto variance = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), sqdiff);
    auto eps      = m.add_literal(migraphx::literal{migraphx::shape{x_type}, {epsilon}});
    auto var_eps  = insert_common_op(m, ins, "add", variance, eps);
    auto rsqrt    = m.insert_instruction(ins, make_op("rsqrt"), var_eps);
    return insert_common_op(m, ins, "mul", x_sub, rsqrt);
}

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
