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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/**
 * If static shape input, creates a literal in migraphx.
 * If dynamic shape input, creates a dimensions_of operator in migraphx (runtime evaluation of
 * shape).
 */
struct parse_shape : op_parser<parse_shape>
{
    std::vector<op_desc> operators() const { return {{"Shape"}}; }

    static std::pair<std::size_t, std::size_t>
    get_range(const shape& input_shape, const onnx_parser::attribute_map& attributes)
    {
        const auto input_ndim = static_cast<int64_t>(input_shape.ndim());
        const auto normalize  = [input_ndim](int64_t axis) {
            axis = std::clamp(axis, -input_ndim, input_ndim);
            return axis < 0 ? axis + input_ndim : axis;
        };
        const auto start =
            normalize(contains(attributes, "start") ? attributes.at("start").i() : 0);
        const auto end =
            normalize(contains(attributes, "end") ? attributes.at("end").i() : input_ndim);
        if(end <= start)
            MIGRAPHX_THROW("PARSE_SHAPE: ending axis <= starting axis, end: " +
                           std::to_string(end) + " start: " + std::to_string(start));
        return {start, end};
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("Shape: operator should have 1 operand");
        const auto& input_shape = args[0]->get_shape();
        const auto [start, end] = get_range(input_shape, info.attributes);

        if(input_shape.dynamic())
        {
            return info.add_instruction(make_op("dimensions_of", {{"start", start}, {"end", end}}),
                                        args[0]);
        }
        else
        {
            std::size_t output_ndim = end - start;
            std::vector<int64_t> vec_shape(output_ndim);
            migraphx::shape s(migraphx::shape::int64_type, {output_ndim});
            std::vector<std::size_t> input_lens = input_shape.lens();
            std::transform(input_lens.begin() + start,
                           input_lens.begin() + end,
                           vec_shape.begin(),
                           [](auto i) { return int64_t(i); });
            return info.add_literal(migraphx::literal{s, vec_shape});
        }
    }

    void infer_symbolic_values(const op_desc&, const symbolic_propagate_context& context) const
    {
        if(context.args.size() != 1)
            return;
        const auto& input_shape = context.args.front()->get_shape();
        if(not can_attach_symbolic_shape(input_shape))
            return;
        const auto [start, end] = get_range(input_shape, context.info.attributes);
        const auto expressions  = input_shape.sym_dims();
        symbolic_tensor_value result{expressions.begin() + start, expressions.begin() + end};
        if(context.output_has_elements(result.size()))
            context.set(std::move(result));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
