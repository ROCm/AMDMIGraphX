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

// Decompose a backward-data convolution (transposed convolution) into a set of stride-1 forward
// convolutions plus an interleave, following MIOpen's implicit-gemm backward-data v4r1 algorithm.
//
// The forward convolution producing y is
//     y[oc, ht] = sum_{ic, j} x[ic, ht*S - P + j*D] * w[oc, ic, j].
// Hence its backward-data, computing dx from dy and w, scatters
//     dx[ic, hi] += dy[oc, ho] * w[oc, ic, j],   with hi = ho*S - P + j*D.
//
// Splitting the filter tap j = idot*Ytilda + itilda  (Ytilda = S/gcd(S,D)) makes
//     hi = S*(ho + idot*(D/g)) + itilda*D - P = S*ht + itilda*D - P,
// so for each residue `itilda` (per spatial dim) the contribution is a plain stride-1
// cross-correlation of dy with the flipped, strided filter slice (dilation D/g), whose output
// `ht` lands on the sub-lattice hi = S*ht + itilda*D - P.  Reassembling the residues = zero-stuff
// each by S, shift by itilda*D, sum (residues occupy disjoint positions), then crop the padding.
namespace {

// Per spatial-dim quantities shared by every residue.
struct dim_info
{
    std::size_t stride;
    std::size_t dilation;
    std::size_t ytilda; // S / gcd(S, D)        -> number of residues along this dim
    std::size_t ydot;   // ceil(Y / ytilda)     -> max taps per residue
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

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// Pad `before`/`after` zeros onto a single axis (a no-op when both are zero).
instruction_ref pad_axis(module& m,
                         instruction_ref ins,
                         instruction_ref t,
                         std::size_t axis,
                         int64_t before,
                         int64_t after)
{
    if(before == 0 and after == 0)
        return t;
    const std::size_t nd = t->get_shape().ndim();
    std::vector<int64_t> pads(2 * nd, 0);
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
    auto u = m.insert_instruction(
        ins, make_op("unsqueeze", {{"axes", {static_cast<int64_t>(axis + 1)}}}), t);
    u         = pad_axis(m, ins, u, axis + 1, 0, static_cast<int64_t>(step - 1));
    auto lens = u->get_shape().lens();
    std::vector<int64_t> rdims(lens.begin(), lens.end());
    rdims[axis] *= static_cast<int64_t>(step);
    rdims.erase(rdims.begin() + axis + 1);
    return m.insert_instruction(ins, make_op("reshape", {{"dims", rdims}}), u);
}

// Per spatial-dim v4r1 quantities derived from the backward-conv attributes and the dy/filter
// sizes.
std::vector<dim_info> compute_dims(const std::vector<std::size_t>& stride,
                                   const std::vector<std::size_t>& dilation,
                                   const std::vector<std::size_t>& dy_lens,
                                   const std::vector<std::size_t>& w_lens)
{
    const std::size_t nsp = stride.size();
    // dy/w carry two leading (batch, channel) axes; their spatial sizes align with stride/dilation.
    auto ho_sp = range(dy_lens.begin() + 2, dy_lens.end());
    auto y_sp  = range(w_lens.begin() + 2, w_lens.end());

    std::vector<dim_info> dims;
    dims.reserve(nsp);
    auto per_dim = views::zip(stride, dilation, ho_sp, y_sp);
    std::transform(per_dim.begin(),
                   per_dim.end(),
                   std::back_inserter(dims),
                   unpack([](auto s, auto dil, auto ho, auto y) {
                       dim_info di;
                       di.stride           = s;
                       di.dilation         = dil;
                       di.y                = y;
                       const std::size_t g = std::gcd(s, dil);
                       di.ytilda           = s / g;
                       di.dd               = dil / g;
                       di.ydot             = ceil_div(y, di.ytilda);
                       di.htilda           = ho + (di.ydot - 1) * di.dd;
                       return di;
                   }));
    return dims;
}

// Build one residue's stride-1 forward convolution: a strided tap-slice of the filter, reshaped
// into forward-conv weight layout (in/out channels swapped) and flipped, then convolved with dy.
instruction_ref make_residue_partial(module& m,
                                     instruction_ref ins,
                                     instruction_ref dy,
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
    std::transform(dim_itilda.begin(),
                   dim_itilda.end(),
                   std::back_inserter(ydot_slice),
                   unpack([](const auto& di, auto it) { return ceil_div(di.y - it, di.ytilda); }));

    // --- weight branch (constant-folded later): strided tap-slice of the filter ---
    auto wsl = w;
    for(std::size_t d = 0; d < nsp; ++d)
    {
        if(itilda[d] > 0) // a slice starting at 0 spans the full axis, so it is a no-op
            wsl = m.insert_instruction(ins,
                                       make_op("slice",
                                               {{"axes", {static_cast<int64_t>(2 + d)}},
                                                {"starts", {static_cast<int64_t>(itilda[d])}},
                                                {"ends", {static_cast<int64_t>(dims[d].y)}}}),
                                       wsl);
        if(dims[d].ytilda > 1)
            wsl = m.insert_instruction(ins,
                                       make_op("step",
                                               {{"axes", {static_cast<int64_t>(2 + d)}},
                                                {"steps", {static_cast<int64_t>(dims[d].ytilda)}}}),
                                       wsl);
    }

    // Reshape the [K, C_pg, *Ydot] backward filter into the forward-conv weight
    // [C_total, K/group, *Ydot] (swap in/out channels, keeping groups intact).
    instruction_ref cw;
    if(group == 1)
    {
        std::vector<int64_t> perm(2 + nsp);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[0], perm[1]);
        cw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), wsl);
    }
    else
    {
        std::vector<int64_t> split_dims;
        split_dims.push_back(group);
        split_dims.push_back(static_cast<int64_t>(k_chan / group));
        split_dims.push_back(static_cast<int64_t>(c_pg));
        for(std::size_t d = 0; d < nsp; ++d)
            split_dims.push_back(static_cast<int64_t>(ydot_slice[d]));
        auto split = m.insert_instruction(ins, make_op("reshape", {{"dims", split_dims}}), wsl);

        std::vector<int64_t> perm(3 + nsp);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[1], perm[2]);
        auto trans =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), split);

        std::vector<int64_t> merge_dims;
        merge_dims.push_back(static_cast<int64_t>(group) * static_cast<int64_t>(c_pg));
        merge_dims.push_back(static_cast<int64_t>(k_chan / group));
        for(std::size_t d = 0; d < nsp; ++d)
            merge_dims.push_back(static_cast<int64_t>(ydot_slice[d]));
        cw = m.insert_instruction(ins, make_op("reshape", {{"dims", merge_dims}}), trans);
    }
    // Flip the filter along the spatial (tap) axes: backward-data is a flipped correlation.
    std::vector<int64_t> rev_axes(nsp);
    std::iota(rev_axes.begin(), rev_axes.end(), static_cast<int64_t>(2));
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
                                dy,
                                cw);
}

// Loop the mixed-radix residue grid, building a partial forward conv per non-empty residue.
residue_set build_partials(module& m,
                           instruction_ref ins,
                           instruction_ref dy,
                           instruction_ref w,
                           const shape& residue_grid,
                           const std::vector<dim_info>& dims,
                           int group,
                           std::size_t k_chan,
                           std::size_t c_pg)
{
    residue_set out;
    for(std::size_t r = 0; r < residue_grid.elements(); ++r)
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
            make_residue_partial(m, ins, dy, w, dims, itilda, group, k_chan, c_pg));
        out.itildas.push_back(itilda);
    }
    return out;
}

// Pixel-shuffle reassembly (no dilation): stack the residues on a new trailing axis, then
// reshape/transpose so residue itilda interleaves into spatial position ht*S + itilda.
instruction_ref reassemble_interleave(module& m,
                                      instruction_ref ins,
                                      const std::vector<instruction_ref>& partials,
                                      const std::vector<dim_info>& dims,
                                      int64_t n_batch,
                                      int64_t c_total)
{
    const std::size_t nsp  = dims.size();
    const int64_t new_axis = 2 + nsp;
    std::vector<instruction_ref> stacked;
    stacked.reserve(partials.size());
    std::transform(partials.begin(), partials.end(), std::back_inserter(stacked), [&](auto p) {
        return m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {new_axis}}}), p);
    });
    auto cat = m.insert_instruction(ins, make_op("concat", {{"axis", new_axis}}), stacked);

    // [N, C, *Htilda, num_res] -> [N, C, *Htilda, *S]
    std::vector<int64_t> split_dims{n_batch, c_total};
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(split_dims),
                   [](const dim_info& di) { return static_cast<int64_t>(di.htilda); });
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(split_dims),
                   [](const dim_info& di) { return static_cast<int64_t>(di.stride); });
    // For 1-D the concat already has this shape (num_res == stride), so the split is a no-op.
    const auto& cat_lens = cat->get_shape().lens();
    auto split =
        (std::equal(split_dims.begin(), split_dims.end(), cat_lens.begin(), cat_lens.end()))
            ? cat
            : m.insert_instruction(ins, make_op("reshape", {{"dims", split_dims}}), cat);
    // interleave each Htilda_d with its stride axis: [N, C, H0, S0, H1, S1, ...]
    std::vector<int64_t> perm{0, 1};
    for(std::size_t d = 0; d < nsp; ++d)
    {
        perm.push_back(static_cast<int64_t>(2 + d));
        perm.push_back(static_cast<int64_t>(2 + nsp + d));
    }
    std::vector<int64_t> identity(perm.size());
    std::iota(identity.begin(), identity.end(), 0);
    auto trans =
        (perm == identity)
            ? split // 1-D: the interleave permutation is a no-op
            : m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), split);
    std::vector<int64_t> merge_dims{n_batch, c_total};
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(merge_dims),
                   [](const dim_info& di) { return static_cast<int64_t>(di.htilda * di.stride); });
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
                       for(std::size_t d = 0; d < nsp; ++d)
                       {
                           const std::size_t axis = 2 + d;
                           partial              = zero_stuff(m, ins, partial, axis, dims[d].stride);
                           const int64_t before = itilda[d] * dims[d].dilation;
                           const int64_t after =
                               (dims[d].ytilda - 1 - itilda[d]) * dims[d].dilation;
                           partial = pad_axis(m, ins, partial, axis, before, after);
                       }
                       return partial;
                   }));
    auto acc = placed.front();
    for(std::size_t i = 1; i < placed.size(); ++i)
        acc = m.insert_instruction(ins, make_op("add"), acc, placed[i]);
    return acc;
}

// Decompose a single convolution_backwards instruction into the v4r1 forward-conv subgraph.
void rewrite_conv_backwards(module& m, instruction_ref ins)
{
    const auto val  = ins->get_operator().to_value();
    auto dy         = ins->inputs().at(0);
    auto w          = ins->inputs().at(1);
    const auto& dys = dy->get_shape();
    const auto& ws  = w->get_shape();
    const auto& os  = ins->get_shape();

    // The decomposition needs concrete spatial/kernel sizes; leave dynamic shapes untouched so the
    // existing (MLIR / reference) path handles them.
    if(dys.dynamic() or ws.dynamic() or os.dynamic())
        return;

    const auto stride   = val.at("stride").to_vector<std::size_t>();
    const auto dilation = val.at("dilation").to_vector<std::size_t>();
    const auto padding  = val.at("padding").to_vector<std::size_t>();
    const int group     = val.at("group").to<int>();

    const std::size_t nsp    = stride.size();
    const auto& dy_lens      = dys.lens();
    const auto& w_lens       = ws.lens();
    const auto& out_lens     = os.lens();
    const std::size_t k_chan = dy_lens.at(1); // dy channels == forward output channels
    const std::size_t c_pg   = w_lens.at(1);  // output channels per group
    if(group <= 0 or k_chan % group != 0)
        return;

    const auto dims = compute_dims(stride, dilation, dy_lens, w_lens);

    // The residues form a mixed-radix grid, one axis per spatial dim with radix ytilda. A shape
    // over those lengths maps each linear residue index `r` back to its per-dim `itilda`.
    std::vector<std::size_t> ytilda_lens(nsp);
    std::transform(dims.begin(), dims.end(), ytilda_lens.begin(), [](const dim_info& di) {
        return di.ytilda;
    });
    const shape residue_grid{shape::uint32_type, ytilda_lens};

    const auto res = build_partials(m, ins, dy, w, residue_grid, dims, group, k_chan, c_pg);

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
        acc = reassemble_interleave(m,
                                    ins,
                                    res.partials,
                                    dims,
                                    static_cast<int64_t>(dy_lens[0]),
                                    static_cast<int64_t>(c_pg) * group);
    else
        acc = reassemble_general(m, ins, res, dims);

    // Crop the padding region to produce dx.
    for(std::size_t d = 0; d < nsp; ++d)
    {
        const int64_t start = padding[d];
        const int64_t end   = start + static_cast<int64_t>(out_lens[2 + d]);
        if(start == 0 and end == static_cast<int64_t>(acc->get_shape().lens()[2 + d]))
            continue; // crop covers the whole axis -> no-op
        acc = m.insert_instruction(
            ins,
            make_op(
                "slice",
                {{"axes", {static_cast<int64_t>(2 + d)}}, {"starts", {start}}, {"ends", {end}}}),
            acc);
    }
    m.replace_instruction(ins, acc);
}

} // namespace

void rewrite_convolution::apply(module& m) const
{
    std::vector<instruction_ref> conv_bwds;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "convolution_backwards")
            conv_bwds.push_back(ins);
    }
    for(auto ins : conv_bwds)
        rewrite_conv_backwards(m, ins);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
