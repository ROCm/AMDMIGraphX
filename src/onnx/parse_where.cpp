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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>
#include <algorithm>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_where : op_parser<parse_where>
{
    std::vector<op_desc> operators() const { return {{"Where"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        const auto s0 = args[0]->get_shape();
        if(shape::same_lens(args[1]->get_shape(), s0) and
           shape::same_lens(args[2]->get_shape(), s0))
            return info.add_instruction(make_op("where"), args[0], args[1], args[2]);

        return migraphx::add_common_op(
            *info.mod, make_op("where"), args, {/*common_type=*/false, /*common_lens=*/true});
    }

    void infer_symbolic_values(const op_desc&, const symbolic_propagate_context& context) const
    {
        const auto condition = context.arg(0);
        const auto x         = context.arg(1);
        const auto y         = context.arg(2);
        if(not condition.has_value() or not x.has_value() or not y.has_value())
            return;
        const auto size = std::max({condition->size(), x->size(), y->size()});
        if(not can_broadcast_value(*condition, size) or not can_broadcast_value(*x, size) or
           not can_broadcast_value(*y, size))
            return;
        const auto condition_values = fixed_integers(*condition);
        if(not condition_values.has_value() or not context.output_has_elements(size))
            return;

        symbolic_tensor_value expressions;
        expressions.reserve(size);
        transform(range(size), std::back_inserter(expressions), [&](auto i) {
            const auto& selected = broadcast_value_at(*condition_values, i) == 0 ? *y : *x;
            return broadcast_value_at(selected, i);
        });
        context.set(std::move(expressions));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
