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
#include <migraphx/rewrite_convolution.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/math.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>
#include <migraphx/zip_view.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Decompose a backward-data convolution (transposed convolution) into a set of stride-1 forward
/// convolutions, following MIOpen's implicit-gemm backward-data v4r1 algorithm.
///
/// For the 1D case, forward convolution producing out is (gather form):
///
///      out[oc, ho] = sum_{ic, j}( inp[ic, ho*S - P + j*D] * w[oc, ic, j] )
///
/// Where S = stride, P = padding, D = dilation, j = kernel spatial index,
/// ic = input channel index, oc = output channel index, ho = output spatial index.
/// sum_{ic, j}() = summation over ic and j.
///
/// Backward-data convolution is (scatter form):
///
///      out[oc, ho] = sum_{ic, j}( inp[ic, hi] * w[ic, oc, j] )
///      ho = hi * S - P + j*D
///
/// Where hi = input spatial index. In this form, the inp/out are the backward convolution's
/// input/output, so the channel and pstial roles and swapped vs. the forward convolution.
///
/// Idea is to split the filter index to do the upsampling as strided access:
///
///      j = idot * ytilda + itilda
///      ytilda = S / gcd(S,D)
///
/// making:
///
///     ho = S*htilda + itilda*D - P    (eqn 1)
///     htilda = hi + idot * (D/g)
///
/// For a set itilda value, note how (eqn 1) becomes constant except for htilda.
/// Continuing with a set itilda value we can define:
///
///     out_itilda[oc, S*htilda + itilda*D - P] := outdot_itilda[oc, htilda]
///
/// where outdot_itilda is a packed matrix. Doing some rearranging, substitutions, and flipping the
/// spatial index we get:
///
///     outdot_itilda[oc, htilda] = sum_{ic, idot}(
///         inp[ic, htilda - (ydot - 1) * (D/g) + idot*(D/g)] * w[ic, oc, (ydot - 1 - idot) * ytilda
///         + itilda]
///     )
///     ydot = ceil(y / ytilda)
///     y = kernel dimension size
///
/// Which is a plain cross-correlation (convolution) with:
///     stride = 1
///     transposed filter with strided access
///     spatial flip
///     dilation = D/g
///     pad = (ydot - 1) * (D/g) [indices beyond the filter are dropped]
///
/// Reassembling the residues for each itilda by zero-stuff each by S, shift by itilda*D,
/// sum (residues occupy disjoint positions), then crop the padding.
namespace {

// Per spatial-dim quantities shared by every residue.
struct dim_info
{
    std::size_t stride;
    std::size_t dilation;
    std::size_t ytilda; // S / gcd(S, D)        -> number of residues along this dim
    std::size_t ydot;   // ceil(y / ytilda)     -> max taps per residue
    std::size_t dd;     // D / gcd(S, D)        -> per-residue conv dilation
    std::size_t htilda; // common per-residue conv output length
    std::size_t y;      // filter length
};

// The partials produced by the residue loop, paired with each residue's itilda (the latter is
// needed only by the general reassembly path).
struct residue_set
{
    std::vector<instruction_ref> partials;
    std::vector<std::vector<std::size_t>> itildas;
};

// Pad `before`/`after` zeros onto a single axis (a no-op when both are zero).
instruction_ref pad_axis(module& m,
                         instruction_ref ins,
                         instruction_ref t,
                         std::size_t axis,
                         std::size_t before,
                         std::size_t after)
{
    if(before == 0 and after == 0)
        return t;
    const std::size_t nd = t->get_shape().ndim();
    std::vector<std::size_t> pads(2 * nd, 0);
    pads[axis]      = before;
    pads[nd + axis] = after;
    return m.insert_instruction(ins, make_op("pad", {{"pads", pads}}), t);
}

// Insert (step-1) zeros after every element along `axis` (spacing the data out by `step`).
instruction_ref
zero_stuff(module& m, instruction_ref ins, instruction_ref t, std::size_t axis, std::size_t step)
{
    if(step <= 1)
        return t;
    auto u     = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {axis + 1}}}), t);
    u          = pad_axis(m, ins, u, axis + 1, 0, step - 1);
    auto rdims = u->get_shape().lens();
    rdims[axis] *= step;
    rdims.erase(rdims.begin() + axis + 1);
    return m.insert_instruction(ins, make_op("reshape", {{"dims", rdims}}), u);
}

// Per spatial-dim v4r1 quantities derived from the backward-conv attributes and the inp/filter
// sizes.
std::vector<dim_info> compute_dims(const std::vector<std::size_t>& stride,
                                   const std::vector<std::size_t>& dilation,
                                   const std::vector<std::size_t>& inp_lens,
                                   const std::vector<std::size_t>& w_lens)
{
    const std::size_t nsp = stride.size();
    // inp/w carry two leading (batch, channel) axes; spatial sizes align with stride/dilation.
    auto hi_sp = range(inp_lens.begin() + 2, inp_lens.end());
    auto y_sp  = range(w_lens.begin() + 2, w_lens.end());

    std::vector<dim_info> dims;
    dims.reserve(nsp);
    auto per_dim = views::zip(stride, dilation, hi_sp, y_sp);
    std::transform(per_dim.begin(),
                   per_dim.end(),
                   std::back_inserter(dims),
                   unpack([](auto s, auto dil, auto hi, auto y) {
                       dim_info di;
                       di.stride           = s;
                       di.dilation         = dil;
                       di.y                = y;
                       const std::size_t g = std::gcd(s, dil);
                       di.ytilda           = s / g;
                       di.dd               = dil / g;
                       di.ydot             = integer_divide_ceil(y, di.ytilda);
                       di.htilda           = hi + (di.ydot - 1) * di.dd;
                       return di;
                   }));
    return dims;
}

// Build one residue's stride-1 forward convolution: a strided tap-slice of the filter, reshaped
// into forward-conv weight layout (in/out channels swapped) and flipped, then convolved with inp.
instruction_ref make_residue_partial(module& m,
                                     instruction_ref ins,
                                     instruction_ref inp,
                                     instruction_ref w,
                                     const std::vector<dim_info>& dims,
                                     const std::vector<std::size_t>& itilda,
                                     int group,
                                     std::size_t k_chan,
                                     std::size_t c_pg)
{
    const std::size_t nsp = dims.size();

    // Number of filter taps kept by this residue along each dim.
    std::vector<std::size_t> ydot_slice;
    ydot_slice.reserve(nsp);
    auto dim_itilda = views::zip(dims, itilda);
    std::transform(
        dim_itilda.begin(),
        dim_itilda.end(),
        std::back_inserter(ydot_slice),
        unpack([](const auto& di, auto it) { return integer_divide_ceil(di.y - it, di.ytilda); }));

    // --- weight branch (constant-folded later): strided tap-slice of the filter ---
    auto wsl       = w;
    const auto idx = range(nsp);
    for(auto&& [d, di, it] : views::zip(idx, dims, itilda))
    {
        if(it > 0) // a slice starting at 0 spans the full axis, so it is a no-op
            wsl = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {2 + d}}, {"starts", {it}}, {"ends", {di.y}}}),
                wsl);
        if(di.ytilda > 1)
            wsl = m.insert_instruction(
                ins, make_op("step", {{"axes", {2 + d}}, {"steps", {di.ytilda}}}), wsl);
    }

    // Reshape the [K, C_pg, *ydot] backward filter into the forward-conv weight
    // [C_total, K/group, *ydot] (swap in/out channels, keeping groups intact).
    instruction_ref cw;
    if(group == 1)
    {
        std::vector<std::size_t> perm(2 + nsp);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[0], perm[1]);
        cw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), wsl);
    }
    else
    {
        const std::size_t groups = group;
        std::vector<std::size_t> split_dims;
        split_dims.push_back(groups);
        split_dims.push_back(k_chan / groups);
        split_dims.push_back(c_pg);
        split_dims.insert(split_dims.end(), ydot_slice.begin(), ydot_slice.end());
        auto split = m.insert_instruction(ins, make_op("reshape", {{"dims", split_dims}}), wsl);

        std::vector<std::size_t> perm(3 + nsp);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[1], perm[2]);
        auto trans =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), split);

        std::vector<std::size_t> merge_dims;
        merge_dims.push_back(groups * c_pg);
        merge_dims.push_back(k_chan / groups);
        merge_dims.insert(merge_dims.end(), ydot_slice.begin(), ydot_slice.end());
        cw = m.insert_instruction(ins, make_op("reshape", {{"dims", merge_dims}}), trans);
    }
    // Flip the filter along the spatial (tap) axes: backward-data is a flipped correlation.
    std::vector<std::size_t> rev_axes(nsp);
    std::iota(rev_axes.begin(), rev_axes.end(), 2);
    cw = m.insert_instruction(ins, make_op("reverse", {{"axes", rev_axes}}), cw);

    // --- the dense, stride-1 forward convolution (one of the v4r1 "gemms") ---
    std::vector<std::size_t> conv_str(nsp, 1);
    std::vector<std::size_t> conv_dil;
    conv_dil.reserve(nsp);
    std::transform(dims.begin(), dims.end(), std::back_inserter(conv_dil), [](const dim_info& di) {
        return di.dd;
    });
    // padding is [leading..., trailing...]: leading spaces the kept taps, trailing pads to htilda.
    std::vector<std::size_t> conv_pad;
    conv_pad.reserve(2 * nsp);
    auto slice_dim = views::zip(ydot_slice, dims);
    std::transform(slice_dim.begin(),
                   slice_dim.end(),
                   std::back_inserter(conv_pad),
                   unpack([](auto slice, const auto& di) { return (slice - 1) * di.dd; }));
    std::transform(dims.begin(), dims.end(), std::back_inserter(conv_pad), [](const dim_info& di) {
        return (di.ydot - 1) * di.dd;
    });
    return m.insert_instruction(ins,
                                make_op("convolution",
                                        {{"padding", conv_pad},
                                         {"stride", conv_str},
                                         {"dilation", conv_dil},
                                         {"group", group}}),
                                inp,
                                cw);
}

// Loop the mixed-radix residue grid, building a partial forward conv per non-empty residue.
residue_set build_partials(module& m,
                           instruction_ref ins,
                           instruction_ref inp,
                           instruction_ref w,
                           const shape& residue_grid,
                           const std::vector<dim_info>& dims,
                           int group,
                           std::size_t k_chan,
                           std::size_t c_pg)
{
    const std::size_t n_residues = residue_grid.elements();
    residue_set out;
    out.partials.reserve(n_residues);
    out.itildas.reserve(n_residues);
    for(std::size_t r : range(n_residues))
    {
        // Decode the linear residue index into itilda per dim.
        auto itilda = residue_grid.multi(r);

        // Skip residues whose first kept tap falls outside the filter (they contribute nothing).
        auto itilda_dim = views::zip(itilda, dims);
        if(std::any_of(itilda_dim.begin(), itilda_dim.end(), unpack([](auto it, const auto& di) {
                           return it >= di.y;
                       })))
            continue;

        out.partials.push_back(
            make_residue_partial(m, ins, inp, w, dims, itilda, group, k_chan, c_pg));
        out.itildas.push_back(itilda);
    }
    return out;
}

// Pixel-shuffle reassembly (no dilation): stack the residues on a new trailing axis, then
// reshape/transpose so residue itilda interleaves into spatial position htilda*S + itilda.
instruction_ref reassemble_interleave(module& m,
                                      instruction_ref ins,
                                      const std::vector<instruction_ref>& partials,
                                      const std::vector<dim_info>& dims,
                                      std::size_t n_batch,
                                      std::size_t c_total)
{
    const std::size_t nsp      = dims.size();
    const std::size_t new_axis = 2 + nsp;
    std::vector<instruction_ref> stacked;
    stacked.reserve(partials.size());
    std::transform(partials.begin(), partials.end(), std::back_inserter(stacked), [&](auto p) {
        return m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {new_axis}}}), p);
    });
    auto cat = m.insert_instruction(ins, make_op("concat", {{"axis", new_axis}}), stacked);

    // [N, C, *Htilda, num_res] -> [N, C, *Htilda, *S]
    std::vector<std::size_t> split_dims{n_batch, c_total};
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(split_dims),
                   [](const dim_info& di) { return di.htilda; });
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(split_dims),
                   [](const dim_info& di) { return di.stride; });
    // For 1-D the concat already has this shape (num_res == stride), so the split is a no-op.
    const auto& cat_lens = cat->get_shape().lens();
    auto split =
        (std::equal(split_dims.begin(), split_dims.end(), cat_lens.begin(), cat_lens.end()))
            ? cat
            : m.insert_instruction(ins, make_op("reshape", {{"dims", split_dims}}), cat);
    // interleave each Htilda_d with its stride axis: [N, C, H0, S0, H1, S1, ...]
    std::vector<std::size_t> perm{0, 1};
    const auto htilda_axes = range(2, 2 + nsp);
    const auto stride_axes = range(2 + nsp, 2 + 2 * nsp);
    for(auto&& [h, s] : views::zip(htilda_axes, stride_axes))
    {
        perm.push_back(h);
        perm.push_back(s);
    }
    std::vector<std::size_t> identity(perm.size());
    std::iota(identity.begin(), identity.end(), 0);
    auto trans =
        (perm == identity)
            ? split // 1-D: the interleave permutation is a no-op
            : m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), split);
    std::vector<std::size_t> merge_dims{n_batch, c_total};
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(merge_dims),
                   [](const dim_info& di) { return di.htilda * di.stride; });
    return m.insert_instruction(ins, make_op("reshape", {{"dims", merge_dims}}), trans);
}

// General reassembly (handles dilation>1 / gcd>1): place each residue onto its sub-lattice via
// zero-stuff by S and shift by itilda*D, then sum (disjoint supports).
instruction_ref reassemble_general(module& m,
                                   instruction_ref ins,
                                   const residue_set& res,
                                   const std::vector<dim_info>& dims)
{
    const std::size_t nsp = dims.size();
    std::vector<instruction_ref> placed;
    placed.reserve(res.partials.size());
    auto partial_itilda = views::zip(res.partials, res.itildas);
    std::transform(partial_itilda.begin(),
                   partial_itilda.end(),
                   std::back_inserter(placed),
                   unpack([&](auto partial, const auto& itilda) {
                       const auto idx = range(nsp);
                       for(auto&& [d, di, it] : views::zip(idx, dims, itilda))
                       {
                           const std::size_t axis   = 2 + d;
                           partial                  = zero_stuff(m, ins, partial, axis, di.stride);
                           const std::size_t before = it * di.dilation;
                           const std::size_t after  = (di.ytilda - 1 - it) * di.dilation;
                           partial = pad_axis(m, ins, partial, axis, before, after);
                       }
                       return partial;
                   }));
    assert(not placed.empty()); // residue 0 (itilda all zero) is never empty, so there is a base
    auto acc = placed.front();
    for(auto p : range(std::next(placed.begin()), placed.end()))
        acc = m.insert_instruction(ins, make_op("add"), acc, p);
    return acc;
}

// Decompose a single convolution_backwards instruction into the v4r1 forward-conv subgraph.
void rewrite_conv_backwards(module& m, instruction_ref ins)
{
    const auto val        = ins->get_operator().to_value();
    auto inp              = ins->inputs().at(0);
    auto w                = ins->inputs().at(1);
    const auto& inp_shape = inp->get_shape();
    const auto& w_shape   = w->get_shape();
    const auto& out_shape = ins->get_shape();

    // The decomposition needs concrete spatial/kernel sizes; leave dynamic shapes untouched so the
    // existing (MLIR / reference) path handles them.
    if(inp_shape.dynamic() or w_shape.dynamic() or out_shape.dynamic())
        return;

    const auto stride   = val.at("stride").to_vector<std::size_t>();
    const auto dilation = val.at("dilation").to_vector<std::size_t>();
    const auto padding  = val.at("padding").to_vector<std::size_t>();
    const int group     = val.at("group").to<int>();

    const std::size_t nsp    = stride.size();
    const auto& inp_lens     = inp_shape.lens();
    const auto& w_lens       = w_shape.lens();
    const auto& out_lens     = out_shape.lens();
    const std::size_t k_chan = inp_lens.at(1); // inp channels == forward output channels
    const std::size_t c_pg   = w_lens.at(1);   // output channels per group
    if(group <= 0 or k_chan % group != 0)
        return;

    // stride/dilation/padding and the tensor spatial axes all share the rank nsp; the zips in
    // compute_dims and the per-dim indexing below assume it.
    assert(dilation.size() == nsp and padding.size() >= nsp and inp_lens.size() == nsp + 2 and
           w_lens.size() == nsp + 2 and out_lens.size() == nsp + 2);

    const auto dims = compute_dims(stride, dilation, inp_lens, w_lens);

    // The residues form a mixed-radix grid, one axis per spatial dim with radix ytilda. A shape
    // over those lengths maps each linear residue index `r` back to its per-dim `itilda`.
    std::vector<std::size_t> ytilda_lens(nsp);
    std::transform(dims.begin(), dims.end(), ytilda_lens.begin(), [](const dim_info& di) {
        return di.ytilda;
    });
    const shape residue_grid{shape::uint32_type, ytilda_lens};

    const auto res = build_partials(m, ins, inp, w, residue_grid, dims, group, k_chan, c_pg);

    // Residue 0 (itilda all zero) is never empty since every filter axis has length >= 1, so the
    // reassembly always has at least one partial to start from.
    assert(not res.partials.empty());

    // With unit stride there is no upsampling, so the single forward convolution is the result.
    // With no dilation every output position is covered exactly once, so the residues reassemble
    // with a pure interleave (concat + reshape/transpose, no pad/add kernels); the `stride <= y`
    // check guarantees no residue is empty, keeping the interleave grid full.
    const bool no_upsample =
        std::all_of(dims.begin(), dims.end(), [](const dim_info& di) { return di.stride == 1; });
    const bool interleave = std::all_of(dims.begin(), dims.end(), [](const dim_info& di) {
        return di.dilation == 1 and di.stride <= di.y;
    });

    instruction_ref acc;
    if(no_upsample)
        acc = res.partials.front();
    else if(interleave)
        acc = reassemble_interleave(m, ins, res.partials, dims, inp_lens[0], c_pg * group);
    else
        acc = reassemble_general(m, ins, res, dims);

    // Crop the padding region to produce out.
    const auto idx = range(nsp);
    for(auto&& [d, pad] : views::zip(idx, padding))
    {
        const std::size_t start = pad;
        const std::size_t end   = start + out_lens[2 + d];
        if(start == 0 and end == acc->get_shape().lens()[2 + d])
            continue; // crop covers the whole axis -> no-op
        acc = m.insert_instruction(
            ins, make_op("slice", {{"axes", {2 + d}}, {"starts", {start}}, {"ends", {end}}}), acc);
    }
    m.replace_instruction(ins, acc);
}

} // namespace

void rewrite_convolution::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution_backwards")
            continue;
        rewrite_conv_backwards(m, ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
