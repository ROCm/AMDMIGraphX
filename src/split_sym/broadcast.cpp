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
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

bool is_broadcast_op(const operation& op)
{
    return contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name());
}

std::vector<shape::dynamic_dimension> symbolic_broadcast_dims(const operation& op)
{
    if(not is_broadcast_op(op))
        return {};
    return from_value<std::vector<shape::dynamic_dimension>>(op.to_value().at("out_dyn_dims"));
}

bool is_symbolic_broadcast(const operation& op, std::size_t ninputs)
{
    if(symbolic_broadcast_dims(op).empty())
        return false;
    if(op.name() == "multibroadcast")
        return ninputs >= 2;
    return ninputs == 2;
}

std::optional<std::vector<input_padding_requirement>>
propagate_padding(instruction_ref ins, const padding_requirement& requirement)
{
    const auto& input_shape  = ins->inputs().front()->get_shape();
    const auto& output_shape = ins->get_shape();
    if(input_shape.ndim() > output_shape.ndim())
        return std::nullopt;

    const auto& op = ins->get_operator();
    std::size_t offset;
    if(op.name() == "broadcast")
        offset = op.to_value().at("axis").to<std::size_t>();
    else
        offset = output_shape.ndim() - input_shape.ndim();

    if(offset > output_shape.ndim() - input_shape.ndim() or requirement.axis < offset or
       requirement.axis >= offset + input_shape.ndim())
        return std::nullopt;
    return propagate_padding_to_input(ins,
                                      0,
                                      requirement.axis - offset,
                                      requirement,
                                      requirement.fills,
                                      requirement.semantics,
                                      requirement.materializable);
}

struct analyze_broadcast : analyzer<analyze_broadcast>
{
    bool matches(const operation& op) const { return is_broadcast_op(op); }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto input_count = ins->inputs().size();
        if(not is_symbolic_broadcast(ins->get_operator(), input_count))
            return symbolic_op_info{ins};
        auto shape_inputs = range(1, input_count);
        auto result =
            analyze_axes(ins, std::vector<std::size_t>{shape_inputs.begin(), shape_inputs.end()});
        result.freezer           = freeze;
        result.propagate_padding = propagate_padding;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const freezer_context& context)
    {
        const auto& op   = source->get_operator();
        auto output_dims = symbolic_broadcast_dims(op);
        std::vector<std::size_t> lens(output_dims.size());
        std::transform(output_dims.begin(), output_dims.end(), lens.begin(), [&](const auto& d) {
            return d.sym_expr.eval_uint(*context.values);
        });
        if(op.name() == "broadcast")
        {
            auto axis = op.to_value().at("axis").to<std::size_t>();
            return m.add_instruction(make_op("broadcast", {{"axis", axis}, {"out_lens", lens}}),
                                     args);
        }
        if(not contains({"multibroadcast", "broadcast_with_dims"}, op.name()))
            MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported symbolic broadcast " + op.name());
        return m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
