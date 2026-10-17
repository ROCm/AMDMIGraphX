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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/sym.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_expand : op_parser<parse_expand>
{
    std::vector<op_desc> operators() const { return {{"Expand"}}; }

    static bool can_resolve_output_dims(const std::vector<shape::dynamic_dimension>& input_dims,
                                        const std::vector<shape::dynamic_dimension>& target_dims)
    {
        const auto& shorter = input_dims.size() < target_dims.size() ? input_dims : target_dims;
        const auto& longer  = input_dims.size() < target_dims.size() ? target_dims : input_dims;
        const auto offset   = longer.size() - shorter.size();
        return std::equal(shorter.begin(),
                          shorter.end(),
                          longer.begin() + offset,
                          [](const auto& input, const auto& target) {
                              return input == target or input == 1 or target == 1 or
                                     input.intersection(target).has_value();
                          });
    }

    static instruction_ref
    add_symbolic_expand(const onnx_parser::node_info& info,
                        const std::vector<instruction_ref>& args,
                        const std::vector<shape::dynamic_dimension>& target_dims)
    {
        const auto input_dims = args[0]->get_shape().to_symbolic().dyn_dims();
        if(not can_resolve_output_dims(input_dims, target_dims))
            return info.add_instruction(make_op("broadcast_with_dims"), args[0], args[1]);
        const auto output_dims = compute_broadcasted_dyn_dims(input_dims, target_dims);
        return info.add_instruction(
            make_op("broadcast_with_dims", {{"out_dyn_dims", to_value(output_dims)}}),
            args[0],
            args[1]);
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        migraphx::argument arg_s = args[1]->eval();
        if(arg_s.empty())
        {
            // variable dims input
            const auto symbolic_dims = args[1]->sym_eval();
            const auto& input_shape  = args[0]->get_shape();
            if(not symbolic_dims.empty() and (not input_shape.dynamic() or input_shape.symbolic()))
            {
                const auto expressions = symbolic_dims.get();
                const std::vector<shape::dynamic_dimension> target_dims(expressions.begin(),
                                                                        expressions.end());
                return add_symbolic_expand(info, args, target_dims);
            }
            return info.add_instruction(make_op("broadcast_with_dims"), args[0], args[1]);
        }
        else
        {
            const shape& shape_0 = args[0]->get_shape();
            std::vector<std::size_t> dims;
            arg_s.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
            if(shape_0.dynamic())
            {
                if(not shape_0.symbolic())
                    MIGRAPHX_THROW(
                        "PARSE_EXPAND: range dynamic input tensor with fixed dims input not "
                        "supported");

                std::vector<sym::expr> expressions;
                std::transform(dims.begin(),
                               dims.end(),
                               std::back_inserter(expressions),
                               [](auto dim) { return sym::lit(dim); });
                const std::vector<shape::dynamic_dimension> target_dims(expressions.begin(),
                                                                        expressions.end());
                return add_symbolic_expand(info, args, target_dims);
            }
            const auto& in_lens = shape_0.lens();
            auto out_lens       = compute_broadcasted_lens(in_lens, dims);
            return info.add_instruction(make_op("multibroadcast", {{"out_lens", out_lens}}),
                                        args[0]);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
