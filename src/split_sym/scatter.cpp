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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <algorithm>
#include <limits>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

instruction_ref
slice_to_lens(module& m, instruction_ref input, const std::vector<std::size_t>& lens)
{
    if(input->get_shape().lens() == lens)
        return input;
    std::vector<int64_t> axes(lens.size());
    std::iota(axes.begin(), axes.end(), int64_t{0});
    std::vector<int64_t> starts(lens.size(), 0);
    std::vector<int64_t> ends(lens.size());
    std::transform(
        lens.begin(), lens.end(), ends.begin(), [](auto x) { return static_cast<int64_t>(x); });
    return m.add_instruction(make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                             input);
}

struct analyze_scatter : analyzer<analyze_scatter>
{
    bool matches(const operation& op) const { return op.name() == "scatter_none"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs = to_shapes(ins->inputs());
        if(inputs.size() != 3)
            return symbolic_op_info{ins};
        const auto& data    = inputs.at(0);
        const auto& indices = inputs.at(1);
        const auto& updates = inputs.at(2);
        auto axis =
            normalize_axis(ins->get_operator().to_value().at("axis").to<int64_t>(), data.ndim());
        if(not axis.has_value() or not shape::is_integral(indices.type()) or
           indices.ndim() != data.ndim() or updates.ndim() != indices.ndim() or
           indices.min_lens() != updates.min_lens() or indices.max_lens() != updates.max_lens() or
           not data.standard() or not has_nonnegative_values(ins->inputs().at(1)))
            return symbolic_op_info{ins};
        if(any_of(range(data.ndim()), [&](std::size_t current_axis) {
               return current_axis != *axis and
                      indices.max_lens().at(current_axis) > data.max_lens().at(current_axis);
           }))
            return symbolic_op_info{ins};

        auto result    = analyze_axes(ins);
        result.freezer = freeze;
        // Index and update extents are equal at runtime but can have independent symbolic roots.
        // Keep this operation isolated to avoid cloning their producer blocks cartesianly.
        result.allow_block_merge = false;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>&)
    {
        assert(args.size() == 3);
        auto data    = args.at(0);
        auto indices = args.at(1);
        auto updates = args.at(2);
        assert(not data->get_shape().dynamic());
        assert(not indices->get_shape().dynamic());
        assert(not updates->get_shape().dynamic());

        const auto& original_indices = source->inputs().at(1)->get_shape();
        auto index_lens              = indices->get_shape().lens();
        auto update_lens             = updates->get_shape().lens();
        std::vector<std::size_t> common_lens(index_lens.size());
        std::transform(index_lens.begin(),
                       index_lens.end(),
                       update_lens.begin(),
                       common_lens.begin(),
                       [](auto x, auto y) { return std::min(x, y); });
        // Independent roots can select different buckets. Their common prefix still contains every
        // valid runtime update because ScatterElements requires indices and updates to have equal
        // runtime shapes.
        if(contains(common_lens, std::size_t{0}))
            return data;
        indices = slice_to_lens(m, indices, common_lens);
        updates = slice_to_lens(m, updates, common_lens);
        if(not updates->get_shape().standard())
            updates = m.add_instruction(make_op("contiguous"), updates);

        std::vector<std::pair<std::size_t, sym::expr>> prefix_axes;
        if(original_indices.symbolic())
            for(std::size_t axis = 0; axis < original_indices.ndim(); ++axis)
                if(is_variable_axis(original_indices.dyn_dims().at(axis)))
                    prefix_axes.emplace_back(axis, original_indices.dyn_dims().at(axis).sym_expr);

        if(prefix_axes.empty())
        {
            if(not indices->get_shape().standard())
                indices = m.add_instruction(make_op("contiguous"), indices);
            return m.add_instruction(source->get_operator(), data, indices, updates);
        }

        auto axis = normalize_axis(source->get_operator().to_value().at("axis").to<int64_t>(),
                                   data->get_shape().ndim());
        assert(axis.has_value());
        auto data_lens = data->get_shape().lens();
        if(data_lens.at(*axis) == std::numeric_limits<std::size_t>::max())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: scatter sink dimension overflows");

        // Zero-padded tail indices would overwrite a valid element zero. Append a disposable sink
        // element and redirect every tail update there instead.
        std::vector<int64_t> pads(data_lens.size() * 2, 0);
        pads.at(data_lens.size() + *axis) = 1;
        auto padded_data = m.add_instruction(make_op("pad", {{"pads", pads}}), data);

        auto valid = make_prefix_mask(m, common_lens, prefix_axes);
        auto sink  = m.add_literal(
            literal{shape{shape::int64_type, {1}}, {static_cast<int64_t>(data_lens.at(*axis))}});
        if(indices->get_shape().type() != shape::int64_type)
            sink = m.add_instruction(
                make_op("convert", {{"target_type", indices->get_shape().type()}}), sink);
        sink    = m.add_instruction(make_op("multibroadcast", {{"out_lens", common_lens}}), sink);
        indices = m.add_instruction(make_op("where"), valid, indices, sink);

        auto scattered = m.add_instruction(source->get_operator(), padded_data, indices, updates);
        auto result =
            m.add_instruction(make_op("slice",
                                      {{"axes", {static_cast<int64_t>(*axis)}},
                                       {"starts", {0}},
                                       {"ends", {static_cast<int64_t>(data_lens.at(*axis))}}}),
                              scattered);
        return m.add_instruction(make_op("contiguous"), result);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
