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
#include <migraphx/rewrite_resize.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/resize.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/*
 * Algorithm of calc_neighbor_points():
 * Input: vvv_ind, a collection of neighbors per resized dimension as:
 *               layer-1: (# resized dimensions, vector)
 *               layer-2: (A vector of 2 of: hi/low)
 *               layer-3: Neighbor index of every pixel in that output dimension (vector)
 *        in_s,  the original input tensor shape (vector)
 *        out_s, the output tensor shape (vector)
 *    resized_m, lens indices that have to resized (map)
 *
 * Output: per resized pixel, its neighboring hi/lo indexes (vector): all permutations.
 * This api stitches all the neighbors (for every dimension) for a resized pixel,
 * to yield its neighbor index w.r.t to the input shape, in_s.
 */
static std::vector<int>
calc_neighbor_points(const std::vector<std::vector<std::vector<std::size_t>>>& vvv_ind,
                     const shape& in_s,
                     const shape& out_s,
                     const std::map<size_t, size_t>& resized_m)
{
    std::size_t ndims       = out_s.ndim();
    const auto& strides     = out_s.strides();
    std::size_t elements_ct = vvv_ind[0][0].size();

    // This function computes for each element, all permutations of its neighbor indices into an
    // Perm block in one go. (Instead of computing each permutation in isolation per element)
    size_t permutations = std::size_t{1} << resized_m.size();
    std::vector<std::vector<std::size_t>> perm_blk(permutations, std::vector<size_t>(strides));

    // final outputted vector: permutations of neighbors.
    std::vector<int> out_idx_vec(permutations * elements_ct);

    for(size_t e_idx = 0; e_idx < elements_ct; ++e_idx)
    {
        size_t t_idx = e_idx;
        for(size_t l_idx = 0; l_idx != ndims; ++l_idx)
        {
            auto entry = resized_m.find(l_idx);
            if(entry != resized_m.end())
            {
                size_t hi_cmp_bit = 1u << entry->second;
                auto lo           = vvv_ind[entry->second][0][e_idx];
                auto hi           = vvv_ind[entry->second][1][e_idx];
                for(size_t i = 0; i < permutations; i++)
                    perm_blk[i][l_idx] = ((i & hi_cmp_bit) != 0) ? hi : lo;
            }
            else
            {
                size_t idx = t_idx / strides[l_idx];
                // no permutations in an unmodified lens index, so idx is copied over:
                for(size_t i = 0; i < permutations; i++)
                    perm_blk[i][l_idx] = idx;
            }
            t_idx %= strides[l_idx];
        }
        // write out the permuted indices, calculated off the perm_blk:
        for(size_t i = 0; i < permutations; i++)
            out_idx_vec[e_idx + elements_ct * i] = in_s.index(perm_blk[i]);
    }
    return out_idx_vec;
}

// Helper to rewrite resize to gather-based implementation for nearest mode
static instruction_ref rewrite_nearest_resize(module& m,
                                              instruction_ref ins,
                                              const shape& in_s,
                                              const std::vector<size_t>& in_lens,
                                              const std::vector<size_t>& out_lens,
                                              const std::vector<float>& scales,
                                              const std::string& nearest_mode,
                                              const std::string& coord_trans_mode)
{
    shape out_s{in_s.type(), out_lens};
    std::size_t out_elements = out_s.elements();
    std::vector<int> ind(out_elements);

    // map out_idx to in_idx
    auto nearest_op = op::resize::get_nearest_op(nearest_mode);
    auto idx_op     = op::resize::get_original_idx_op(coord_trans_mode);

    shape_for_each(out_s, [&](const auto& out_idx_v, size_t out_idx) {
        std::vector<size_t> in_idx(out_idx_v.size());
        for(std::size_t ii = 0; ii < in_lens.size(); ++ii)
        {
            auto idx_val = idx_op(in_lens[ii], out_lens[ii], out_idx_v[ii], scales[ii]);
            in_idx[ii]   = nearest_op(in_lens[ii], idx_val);
        }

        ind[out_idx] = static_cast<int64_t>(in_s.index(in_idx));
    });

    // reshape input to one-dimension
    std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
    auto rsp =
        m.insert_instruction(ins, make_op("reshape", {{"dims", rsp_lens}}), ins->inputs()[0]);

    // ins_ind should be a multi dimensional index that will restore original rank
    shape ind_s{shape::int32_type, out_lens};
    auto ins_ind = m.add_literal(literal(ind_s, ind));
    return m.replace_instruction(ins, make_op("gather", {{"axis", 0}}), rsp, ins_ind);
}

// Helper to rewrite resize to gather-based implementation for linear mode
static instruction_ref rewrite_linear_resize(module& m,
                                             instruction_ref ins,
                                             const shape& in_s,
                                             const std::vector<size_t>& in_lens,
                                             const std::vector<size_t>& out_lens,
                                             const std::vector<float>& scales,
                                             const std::string& coord_trans_mode)
{
    shape out_s{in_s.type(), out_lens};

    // reshape input to one-dimension
    std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
    auto rsp =
        m.insert_instruction(ins, make_op("reshape", {{"dims", rsp_lens}}), ins->inputs()[0]);

    auto nearest_floor = op::resize::get_nearest_op("floor");
    auto nearest_ceil  = op::resize::get_nearest_op("ceil");

    std::vector<size_t> resized_axes; // vector of dimensions to be resized
    std::size_t out_elements = 1;     // total number of elements to be resized
    size_t resized_ct        = 0;
    std::map<size_t, size_t> resized_m; // modified indices --> vvv_ind index below
    for(std::size_t axis = 0; axis != out_lens.size(); ++axis)
    {
        out_elements *= out_lens[axis];
        if(in_lens[axis] == out_lens[axis])
            continue;
        resized_axes.push_back(axis);
        resized_m[axis] = resized_ct++;
    }

    // Neighbor indices. For an axis. Two sets of max/min per element:
    std::vector<std::vector<std::size_t>> vv_ind(2, std::vector<std::size_t>(out_elements));
    // Neighbor indices. For all resized axes:
    std::vector<std::vector<std::vector<std::size_t>>> vvv_ind(resized_ct, vv_ind);
    // Delta list. For each resized axes - per element.
    std::vector<std::vector<float>> delta(resized_ct, std::vector<float>(out_elements));

    auto idx_op = op::resize::get_original_idx_op(coord_trans_mode);
    shape_for_each(out_s, [&](const auto& out_idx_v, std::size_t out_idx) {
        for(size_t ii = 0; ii != resized_ct; ++ii)
        {
            auto idx     = resized_axes[ii];
            auto idx_val = idx_op(in_lens[idx], out_lens[idx], out_idx_v[idx], scales[idx]);
            vvv_ind[ii][0][out_idx] = nearest_floor(in_lens[idx], idx_val);
            vvv_ind[ii][1][out_idx] = nearest_ceil(in_lens[idx], idx_val);
            delta[ii][out_idx]      = idx_val - vvv_ind[ii][0][out_idx];
        }
    });

    auto ind = calc_neighbor_points(vvv_ind, in_s.as_standard(), out_s, resized_m);

    auto dim_lens = out_lens;
    // indices matrix size grows 2x per resized-axis:
    dim_lens[0] *= (1u << resized_ct);
    shape ind_s{shape::int32_type, dim_lens};
    auto ins_ind = m.add_literal(literal(ind_s, ind));
    auto data    = m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), rsp, ins_ind);

    for(auto idx = resized_ct; idx != 0u; --idx)
    {
        dim_lens[0] /= 2; // halved for 2 slices of data (hi & low below)
        shape dim_s{in_s.type(), dim_lens};
        const auto& dim_delta = delta[idx - 1];
        std::vector<float> delta_data;
        for(std::size_t j = 0; j < dim_lens[0] / out_lens[0]; ++j)
            delta_data.insert(delta_data.begin(), dim_delta.begin(), dim_delta.end());
        auto ins_delta = m.add_literal(dim_s, delta_data);

        // slice the data
        int64_t slc_stride = dim_lens[0];
        auto low           = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {slc_stride}}}), data);
        auto hi = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {slc_stride}}, {"ends", {2 * slc_stride}}}),
            data);
        auto diff = m.insert_instruction(ins, make_op("sub"), hi, low);
        auto ddf  = m.insert_instruction(ins, make_op("mul"), diff, ins_delta);
        data      = m.insert_instruction(ins, make_op("add"), ddf, low);
    }
    return m.replace_instruction(ins, data);
}

static bool is_affine_resize(const op::resize& rop,
                             const std::vector<std::size_t>& in_lens,
                             const std::vector<std::size_t>& out_lens)
{
    if(not migraphx::equal(in_lens, out_lens, [&](auto in_len, auto out_len) {
           return (std::max(in_len, out_len) % std::min(in_len, out_len)) == 0;
       }))
        return false;
    if(rop.mode == "nearest")
        return true;
    if(rop.mode != "linear")
        return false;
    return migraphx::equal(in_lens, out_lens, std::greater_equal<>{});
}

void rewrite_resize::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "resize")
            continue;

        if(ins->get_shape().ndim() >= 64)
            continue;

        // Skip if input has dynamic shape
        if(ins->inputs().empty() or ins->inputs()[0]->get_shape().dynamic())
            continue;

        // Only handle 1-input mode (scales/sizes as attributes)
        if(ins->inputs().size() != 1)
            continue;

        auto resize_op = any_cast<op::resize>(ins->get_operator());
        auto in_s      = ins->inputs()[0]->get_shape();
        auto in_lens   = in_s.lens();
        auto out_lens  = ins->get_shape().lens();

        if(affine_only and not is_affine_resize(resize_op, in_lens, out_lens))
            continue;

        std::vector<float> scales = resize_op.scales;
        if(scales.empty())
        {
            scales.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           scales.begin(),
                           [](float in, float out) { return out / in; });
        }

        if(std::all_of(
               scales.begin(), scales.end(), [](auto scale) { return float_equal(scale, 1.0f); }))
        {
            m.replace_instruction(ins, ins->inputs()[0]);
        }
        else if(resize_op.mode == "nearest")
        {
            rewrite_nearest_resize(m,
                                   ins,
                                   in_s,
                                   in_lens,
                                   out_lens,
                                   scales,
                                   resize_op.nearest_mode,
                                   resize_op.coordinate_transformation_mode);
        }
        else if(resize_op.mode == "linear")
        {
            rewrite_linear_resize(
                m, ins, in_s, in_lens, out_lens, scales, resize_op.coordinate_transformation_mode);
        }
        // Other modes (cubic, etc.) are not yet supported for rewriting
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
