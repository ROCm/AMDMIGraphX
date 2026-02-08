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
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/transform_view.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <bitset>
#include <map>
#include <numeric>

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
static std::vector<std::size_t>
calc_neighbor_points(const std::vector<std::vector<std::vector<std::size_t>>>& vvv_ind,
                     const shape& in_s,
                     const shape& out_s,
                     const std::map<size_t, size_t>& resized_m)
{
    assert(resized_m.size() < 64);
    const std::size_t ndims       = out_s.ndim();
    const std::size_t elements_ct = out_s.elements();

    // This function computes for each element, all permutations of its neighbor indices.
    // Each permutation selects hi or lo for each resized dimension based on bit patterns.
    const size_t permutations = std::size_t{1} << resized_m.size();

    // 2D shape to decompose flat index into [perm_idx, e_idx]
    const shape flat_s{shape::uint64_type, {permutations, elements_ct}};

    // final outputted vector: permutations of neighbors.
    std::vector<std::size_t> out_idx_vec(permutations * elements_ct);

    // Compute all permutations x elements in a single transform.
    // Output layout: [perm_idx * elements_ct + e_idx].
    auto indices = range(out_idx_vec.size());
    par_transform(indices.begin(),
                  indices.end(),
                  out_idx_vec.begin(),
                  [&](std::ptrdiff_t flat_idx) -> std::size_t {
                      // TODO: Use stuctured binding when switching to C++23
                      auto multi_idx = flat_s.multi<2>(flat_idx);
                      auto perm_idx  = multi_idx[0];
                      auto e_idx     = multi_idx[1];

                      std::bitset<64> bits(perm_idx);
                      auto out_idx_v = out_s.multi<64>(e_idx);

                      std::array<std::size_t, 64> neighbor_idx{};
                      // Build multi-dimensional neighbor index for this permutation
                      auto dim_indices = range(ndims);
                      std::transform(dim_indices.begin(),
                                     dim_indices.end(),
                                     neighbor_idx.begin(),
                                     [&](std::ptrdiff_t dim) -> size_t {
                                         auto entry = resized_m.find(dim);
                                         if(entry != resized_m.end())
                                         {
                                             auto bit_pos = entry->second;
                                             return bits.test(bit_pos) ? vvv_ind[bit_pos][1][e_idx]
                                                                       : vvv_ind[bit_pos][0][e_idx];
                                         }
                                         return out_idx_v[dim];
                                     });
                      return in_s.index(neighbor_idx.begin(), neighbor_idx.begin() + ndims);
                  });
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
    std::vector<std::size_t> ind(out_elements);

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

        ind[out_idx] = in_s.index(in_idx);
    });

    auto rsp = m.insert_instruction(
        ins, make_op("reshape", {{"dims", {in_s.elements()}}}), ins->inputs()[0]);

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

    std::size_t out_elements = out_s.elements();

    std::vector<size_t> resized_axes; // vector of dimensions to be resized
    auto axes = range(out_lens.size());
    std::copy_if(axes.begin(),
                 axes.end(),
                 std::back_inserter(resized_axes),
                 [&](std::ptrdiff_t axis) { return not float_equal(scales[axis], 1.0f); });

    size_t resized_ct = resized_axes.size();
    std::map<size_t, size_t> resized_m; // modified indices --> vvv_ind index below
    auto ra_indices = range(resized_ct);
    std::transform(resized_axes.begin(),
                   resized_axes.end(),
                   ra_indices.begin(),
                   std::inserter(resized_m, resized_m.end()),
                   [&](std::size_t ra, std::size_t i) { return std::make_pair(ra, i); });

    // Neighbor indices. For an axis. Two sets of max/min per element:
    std::vector<std::vector<std::size_t>> vv_ind(2, std::vector<std::size_t>(out_elements));
    // Neighbor indices. For all resized axes:
    std::vector<std::vector<std::vector<std::size_t>>> vvv_ind(resized_ct, vv_ind);
    // Delta list. For each resized axes - per element.
    std::vector<std::vector<float>> delta(resized_ct, std::vector<float>(out_elements));

    auto idx_op = op::resize::get_original_idx_op(coord_trans_mode);
    par_for(out_elements, [&](std::size_t out_idx) {
        auto multi_out_idx = out_s.multi<64>(out_idx);
        for(size_t ii : range(resized_axes.size()))
        {
            auto idx     = resized_axes[ii];
            auto idx_val = idx_op(in_lens[idx], out_lens[idx], multi_out_idx[idx], scales[idx]);
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
    auto ins_ind    = m.add_literal(literal(ind_s, ind));
    auto gather_ins = m.insert_instruction(ins, make_op("gather", {{"axis", 0}}), rsp, ins_ind);

    // Lazy views for per-axis tile scales (1, 2, 4, ...)
    auto scale_indices = range(resized_ct);
    auto tile_scales =
        views::transform(scale_indices, [](std::size_t i) -> std::size_t { return 1ull << i; });

    // Build delta literals from tile scales and per-axis deltas
    std::vector<instruction_ref> ins_deltas(resized_ct);
    std::transform(tile_scales.begin(),
                   tile_scales.end(),
                   delta.begin(),
                   ins_deltas.begin(),
                   [&](std::size_t scale, const std::vector<float>& dim_delta) {
                       shape tile_s{shape::uint64_type, {scale, dim_delta.size()}};
                       std::vector<float> delta_data(tile_s.elements());
                       auto rep_indices = range(delta_data.size());
                       std::transform(
                           rep_indices.begin(),
                           rep_indices.end(),
                           delta_data.begin(),
                           [&](std::ptrdiff_t j) { return dim_delta[tile_s.multi<2>(j)[1]]; });

                       auto step_lens = out_lens;
                       step_lens[0] *= scale;
                       shape dim_s{in_s.type(), step_lens};
                       return m.add_literal(dim_s, delta_data);
                   });

    // Fold in reverse order: largest scale first
    instruction_ref data = std::inner_product(
        ins_deltas.rbegin(),
        ins_deltas.rend(),
        std::make_reverse_iterator(tile_scales.end()),
        gather_ins,
        [&](instruction_ref start, auto p) {
            auto [ins_delta, scale] = p;
            auto slc_stride         = out_lens[0] * scale;
            auto low                = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {slc_stride}}}),
                start);
            auto hi = m.insert_instruction(
                ins,
                make_op("slice",
                        {{"axes", {0}}, {"starts", {slc_stride}}, {"ends", {2 * slc_stride}}}),
                start);
            auto diff = m.insert_instruction(ins, make_op("sub"), hi, low);
            auto ddf  = m.insert_instruction(ins, make_op("mul"), diff, ins_delta);
            return m.insert_instruction(ins, make_op("add"), ddf, low);
        },
        MIGRAPHX_LIFT(std::make_pair));
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
