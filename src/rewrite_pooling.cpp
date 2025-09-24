/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/op/lrn.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void replace_with_reduce(module& m, instruction_ref ins)
{
    auto&& s  = ins->inputs().front()->get_shape();
    auto&& op = any_cast<op::pooling>(ins->get_operator());
    auto lens = s.lens();
    std::vector<std::int64_t> axes(lens.size() - 2);
    std::iota(axes.begin(), axes.end(), 2);

    // average pooling
    if(op.mode == op::pooling_mode::average)
    {
        m.replace_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), ins->inputs());
    }
    // max pooling
    else
    {
        m.replace_instruction(ins, make_op("reduce_max", {{"axes", axes}}), ins->inputs());
    }
}

static void lower_lrn_to_pooling(module& m, instruction_ref ins)
{
    auto v = ins->get_operator().to_value();

    float alpha = v.at("alpha").to<float>();
    float beta  = v.at("beta").to<float>();
    float k     = v.at("bias").to<float>();
    int   size  = v.at("size").to<int>();
    const unsigned int axis = 1;

    auto x = ins->inputs().at(0);
    const auto& xshape = x->get_shape();
    auto lens = xshape.lens();
    const int64_t rank = static_cast<int64_t>(lens.size());
    if(rank < 2 or axis >= rank) return;
    if(size <= 0) return;

    auto x2 = m.insert_instruction(ins, make_op("mul"), x, x);

    std::vector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[static_cast<std::size_t>(caxis)], perm.back());
    auto moved = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), x2);
    auto moved_lens = moved->get_shape().lens();

    auto b = std::accumulate(moved_lens.begin(), moved_lens.end() - 1, 1, std::multiplies<size_t>());
    const int64_t c = static_cast<int64_t>(moved_lens.back());
    auto pooled_in = m.insert_instruction(
        ins, 
        make_op("reshape", {{"dims", std::vector<int64_t>{static_cast<int64_t>(b), 1, 1, c}}}),
        moved);

    auto avg = m.insert_instruction(
        ins,
        make_op("pooling",
                {{"mode", op::pooling_mode::average},
                 {"lengths", std::vector<int64_t>{1, size}},
                 {"stride",  std::vector<int64_t>{1, 1}},
                 {"padding", std::vector<int64_t>{0, size/2}},
                 {"count_include_pad", true}}),
        pooled_in);

    auto moved_shape_back =
        std::vector<int64_t>(moved_lens.begin(), moved_lens.end());
    auto avg_moved = m.insert_instruction(
        ins, make_op("reshape", {{"dims", moved_shape_back}}), avg);


    auto invp = invert_permutation(perm);
    auto avg_ch = m.insert_instruction(ins, make_op("transpose", {{"permutation", invp}}), avg_moved);

    auto k_lit   = m.add_literal(k);
    auto a_lit   = m.add_literal(alpha);
    auto k_mb    = m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), k_lit);
    auto a_mb    = m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), a_lit);
    auto alpha_avg = m.insert_instruction(ins, make_op("mul"), a_mb, avg_ch);
    auto den       = m.insert_instruction(ins, make_op("add"), k_mb, alpha_avg);

    auto b_lit  = m.add_literal(beta);
    auto b_mb   = m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), b_lit);
    auto denpow = m.insert_instruction(ins, make_op("pow"), den, b_mb);
    auto y      = m.insert_instruction(ins, make_op("div"), ins->inputs().front(), denpow);

    m.replace_instruction(ins, y);
}



static void replace_dilations_with_gather_pooling(module& m, instruction_ref ins)
{
    // TODO remove this when MIOpen supports dilated pooling
    auto&& s  = ins->inputs().front()->get_shape();
    auto&& op = any_cast<op::pooling>(ins->get_operator());
    // Ignore N, C axes
    std::vector<size_t> dims = {s.lens().cbegin() + 2, s.lens().cend()};

    bool default_padding =
        std::all_of(op.padding.cbegin(), op.padding.cend(), [](auto i) { return i == 0; });

    if(not default_padding)
    {
        for(size_t idx{0}; idx < op.padding.size(); ++idx)
        {
            // We need to pad both ends
            dims[idx] += op.padding.at(idx) * 2;
        }
    }
    std::vector<size_t> kernels   = op.lengths;
    std::vector<size_t> strides   = op.stride;
    std::vector<size_t> dilations = op.dilations;

    std::vector<std::vector<int>> axis_indices;
    axis_indices.resize(dims.size());

    for(auto idx{0}; idx < dims.size(); ++idx)
    {
        // Only consider if iw fits into the window
        for(size_t stride{0}; stride < dims.at(idx) - dilations.at(idx) * (kernels.at(idx) - 1);
            stride += strides.at(idx))
        {
            for(size_t step{0}; step < kernels.at(idx); ++step)
            {
                axis_indices.at(idx).push_back(stride + dilations.at(idx) * step);
            }
        }
    }

    auto elements = ins->inputs().front();
    if(not default_padding)
    {
        // Pad supports asym, we need to provide both ends
        std::vector<size_t> padding(2 * s.lens().size(), 0);
        // Format will be e.g {N, C, P1, P2, N, C, P1, P2}
        for(size_t idx{0}; idx < op.padding.size(); ++idx)
        {
            // Ignore N, C axes
            padding.at(2 + idx)                   = op.padding.at(idx);
            padding.at(2 + idx + s.lens().size()) = op.padding.at(idx);
        }

        // Default value needed for Max pooling
        elements = m.insert_instruction(
            ins,
            make_op("pad", {{"pads", padding}, {"value", std::numeric_limits<float>::lowest()}}),
            elements);
    }

    for(auto idx{0}; idx < axis_indices.size(); ++idx)
    {
        migraphx::shape s_indices{migraphx::shape::int32_type, {axis_indices.at(idx).size()}};
        auto indices = m.add_literal(migraphx::literal{s_indices, axis_indices.at(idx)});
        elements     = m.insert_instruction(
            ins, make_op("gather", {{"axis", idx + 2 /*ignore N,C*/}}), elements, indices);
    }

    // Ignore padding
    std::vector<size_t> new_padding(kernels.size(), 0);
    // The kernel window elements are places next to each other. E.g. {x1, y1, x2, y2, ...}
    // We need to skip them to not overlap
    std::vector<size_t> new_strides(kernels);
    // Ignore dilations
    std::vector<size_t> new_dilations(kernels.size(), 1);
    m.replace_instruction(ins,
                          make_op("pooling",
                                  {{"mode", op.mode},
                                   {"padding", new_padding},
                                   {"stride", new_strides},
                                   {"lengths", kernels},
                                   {"dilations", new_dilations}}),
                          elements);
}

void rewrite_pooling::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->inputs().empty())
            continue;  
        if(ins->name() == "lrn")
        {
            lower_lrn_to_pooling(m, ins);
            continue;
        }
        if(ins->name() != "pooling")
            continue;

        auto&& s                  = ins->inputs().front()->get_shape();
        auto&& op                 = any_cast<op::pooling>(ins->get_operator());
        bool same_kernel_as_shape = std::equal(
            s.lens().cbegin() + 2, s.lens().cend(), op.lengths.cbegin(), op.lengths.cend());
        bool default_strides =
            std::all_of(op.stride.cbegin(), op.stride.cend(), [](auto i) { return i == 1; });
        bool default_padding =
            std::all_of(op.padding.cbegin(), op.padding.cend(), [](auto i) { return i == 0; });
        bool default_dilations =
            std::all_of(op.dilations.cbegin(), op.dilations.cend(), [](auto i) { return i == 1; });
        if(same_kernel_as_shape and default_strides and default_padding and default_dilations)
        {
            replace_with_reduce(m, ins);
        }
        else if(not default_dilations)
        {
            // Dilated AvgPool with padding is not supported
            if(not default_padding and op.mode == op::pooling_mode::average)
            {
                continue;
            }
            auto size =
                std::accumulate(s.lens().cbegin(), s.lens().cend(), 1, std::multiplies<size_t>());
            // Can't handle too much size because of literal size
            if(size > 100000)
            {
                continue;
            }

            replace_dilations_with_gather_pooling(m, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
