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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_DIM_OPS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_DIM_OPS_HPP

#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>

#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Reads a dim that has to be known while building, so that it can be used in arithmetic rather
/// than emitted into the graph. `what` names the operator and axis for the error message.
inline std::size_t static_dim(const shape& s, std::size_t axis, const std::string& what)
{
    if(s.dynamic() and not s.dyn_dims().at(axis).is_fixed())
        MIGRAPHX_THROW(what + " must be a fixed dimension, but got shape " + to_string(s));
    return s.max_lens().at(axis);
}

/// Shape-changing operators whose target dims may be partly symbolic. Each keeps the static
/// attribute while every dim is known, so a graph that happens to be fully determined emits the
/// same operator it would have without any symbolic dims in play.

inline operation make_multibroadcast(const std::vector<sym::expr>& dims)
{
    const auto entries = to_dim_like(dims);
    if(all_ints(entries))
        return make_op("multibroadcast", {{"out_lens", to_ints(entries)}});
    std::vector<shape::dynamic_dimension> dyn_dims(dims.size());
    std::transform(dims.begin(), dims.end(), dyn_dims.begin(), [](const sym::expr& e) {
        return shape::dynamic_dimension{e};
    });
    return make_op("multibroadcast", {{"out_dyn_dims", to_value(dyn_dims)}});
}

inline operation make_reshape(const std::vector<sym::expr>& dims)
{
    return make_op("reshape", {{"dims", to_value(to_dim_like(dims))}});
}

/// Counts up from zero along axis `length_axis` of `out_dims`, leaving every other axis at one so
/// the result broadcasts. A known length becomes a literal; otherwise the length is read at
/// runtime from axis `shape_axis` of `length_of`, which folds back to that same literal once the
/// length has been specialized.
inline instruction_ref insert_iota(module& m,
                                   instruction_ref ins,
                                   const std::vector<sym::expr>& out_dims,
                                   std::size_t length_axis,
                                   instruction_ref length_of,
                                   std::size_t shape_axis,
                                   shape::type_t type)
{
    const auto entries = to_dim_like(out_dims);
    if(all_ints(entries))
    {
        const auto lens = to_ints(entries);
        std::vector<int64_t> range(lens.at(length_axis));
        std::iota(range.begin(), range.end(), 0);
        return m.add_literal(
            literal{shape{type, std::vector<std::size_t>(lens.begin(), lens.end())}, range});
    }
    auto len = m.insert_instruction(
        ins, make_op("dimensions_of", {{"start", shape_axis}, {"end", shape_axis + 1}}), length_of);
    const auto count_type = len->get_shape().type();
    auto zero             = m.add_literal(literal{shape{count_type, {1}}, {0}});
    auto one              = m.add_literal(literal{shape{count_type, {1}}, {1}});
    auto range            = m.insert_instruction(
        ins,
        make_op("dynamic_range",
                           {{"output_dim", to_value(shape::dynamic_dimension{out_dims.at(length_axis)})}}),
        zero,
        len,
        one);
    range = m.insert_instruction(ins, make_op("convert", {{"target_type", type}}), range);
    return m.insert_instruction(ins, make_reshape(out_dims), range);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
