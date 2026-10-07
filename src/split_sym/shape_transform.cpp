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
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

operation reshape_from_shape(const shape& target,
                             const std::unordered_map<sym::expr, sym::expr>& substitutions)
{
    std::vector<dim_like> dims(target.ndim());
    std::transform(
        target.dyn_dims().begin(), target.dyn_dims().end(), dims.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    return make_op("reshape", {{"dims", to_value(dims)}});
}

std::vector<sym::expr> non_unit_dims(const shape& s)
{
    std::vector<sym::expr> result;
    std::transform(s.dyn_dims().begin(),
                   s.dyn_dims().end(),
                   std::back_inserter(result),
                   [](const auto& d) { return d.sym_expr; });
    result.erase(std::remove(result.begin(), result.end(), sym::lit(1)), result.end());
    return result;
}

struct analyze_shape_transform : analyzer<analyze_shape_transform>
{
    bool matches(const operation& op) const
    {
        return contains({"contiguous", "flatten", "reshape", "transpose"}, op.name());
    }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& op     = ins->get_operator();
        auto inputs        = to_shapes(ins->inputs());
        auto descriptor_op = op;
        const auto& output = ins->get_shape();
        std::vector<std::size_t> shape_input_indices;
        bool needs_freezer = false;
        if(op.name() == "reshape")
        {
            if(inputs.size() == 2 and inputs.back().symbolic())
            {
                auto target_reshape = reshape_from_shape(inputs.back(), {});
                descriptor_op = make_op("reshape", {{"dims", to_value(inputs.back().max_lens())}});
                auto input_elements  = inputs.front().sym_elements();
                auto output_elements = output.sym_elements();
                if(sym::strict_less(input_elements, output_elements).value_or(false) or
                   sym::strict_less(output_elements, input_elements).value_or(false))
                    return symbolic_op_info{ins};
                if(target_reshape.compute_shape({inputs.front()}) != output)
                    return symbolic_op_info{ins};
                shape_input_indices = {1};
                needs_freezer       = true;
            }
            else if(inputs.size() != 1)
                return symbolic_op_info{ins};

            if(non_unit_dims(inputs.front()) == non_unit_dims(output))
            {
                auto result = analyze_axes(ins, std::move(shape_input_indices));
                if(needs_freezer)
                    result.freezer = freeze;
                return result;
            }
        }
        else if(inputs.size() != 1)
            return symbolic_op_info{ins};
        auto desc = shape_transform_descriptor::create(inputs.front().max_lens(), {descriptor_op});
        if(desc.empty())
            return symbolic_op_info{ins};
        auto source_dims = inputs.front().to_symbolic().dyn_dims();
        auto output_dims = output.to_symbolic().dyn_dims();
        auto result      = analyze_axes(
            ins,
            std::move(shape_input_indices),
            [&](std::size_t axis) {
                auto source_axes = range(source_dims.size());
                auto count =
                    std::count_if(source_axes.begin(), source_axes.end(), [&](auto source_axis) {
                        auto dst_axes = desc.get_dst_axes_from_src(source_axis);
                        return dst_axes.size() == 1 and dst_axes.front() == axis and
                               source_dims.at(source_axis).sym_expr ==
                                   output_dims.at(axis).sym_expr;
                    });
                return count == 1;
            },
            [&](std::size_t, std::size_t axis) {
                auto dst_axes = desc.get_dst_axes_from_src(axis);
                if(dst_axes.size() != 1)
                    return axis_desc{};
                auto dst_axis = dst_axes.front();
                return source_dims.at(axis).sym_expr == output_dims.at(dst_axis).sym_expr
                           ? parallel_axis()
                           : axis_desc{};
            });
        if(needs_freezer)
            result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>& values)
    {
        assert(source->inputs().size() == 2);
        assert(args.size() == 1);
        const auto& target = source->inputs().back()->get_shape();
        std::vector<int64_t> dims(target.ndim());
        std::transform(
            target.dyn_dims().begin(), target.dyn_dims().end(), dims.begin(), [&](const auto& d) {
                return static_cast<int64_t>(d.sym_expr.eval_uint(values));
            });
        return m.add_instruction(make_op("reshape", {{"dims", dims}}), args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
