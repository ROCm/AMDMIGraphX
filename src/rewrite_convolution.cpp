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
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
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

// A value per spatial dim of the convolution that computes every residue: spatial(di) for
// each dim, followed by residue(di) for the dims that have more than one residue
template <class Spatial, class Residue>
std::vector<std::size_t>
residue_conv_dims(const std::vector<dim_info>& dims, Spatial spatial, Residue residue)
{
    return std::accumulate(
        dims.begin(), dims.end(), std::vector<std::size_t>{}, [&](auto result, const dim_info& di) {
            result.push_back(spatial(di));
            if(di.stride > 1)
                result.push_back(residue(di));
            return result;
        });
}

// With no dilation the residues are computed by one convolution that has a spatial dim per
// residue axis where the input has one element (see find_convolution_pixel_shuffle), and that
// writes them interleaved. Tap j of the filter belongs to residue j % S and is tap j / S of its
// convolution, so the flipped filter of ydot * S taps splits into (ydot, S).
instruction_ref residue_convolution(module& m,
                                    instruction_ref ins,
                                    instruction_ref inp,
                                    instruction_ref w,
                                    const std::vector<dim_info>& dims)
{
    const std::size_t ndim = 2 + dims.size();
    const auto& inp_lens   = inp->get_shape().lens();
    const auto one         = [](const dim_info&) -> std::size_t { return 1; };

    std::vector<std::size_t> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[0], perm[1]);
    auto cw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), w);
    std::vector<std::size_t> axes(dims.size());
    std::iota(axes.begin(), axes.end(), 2);
    cw = m.insert_instruction(ins, make_op("reverse", {{"axes", axes}}), cw);
    std::vector<std::size_t> w_dims = {w->get_shape().lens()[1], inp_lens[1]};
    const auto taps                 = residue_conv_dims(
        dims,
        [](const dim_info& di) { return di.ydot; },
        [](const dim_info& di) { return di.stride; });
    w_dims.insert(w_dims.end(), taps.begin(), taps.end());
    cw = m.insert_instruction(ins, make_op("reshape", {{"dims", w_dims}}), cw);

    // A reshape, unlike an unsqueeze, gives the unit dims strides that follow the layout of inp
    std::vector<std::size_t> x_dims = {inp_lens[0], inp_lens[1]};
    const auto x_spatial =
        residue_conv_dims(dims, [](const dim_info& di) { return di.htilda + 1 - di.ydot; }, one);
    x_dims.insert(x_dims.end(), x_spatial.begin(), x_spatial.end());
    auto x = m.insert_instruction(ins, make_op("reshape", {{"dims", x_dims}}), inp);

    auto padding = residue_conv_dims(
        dims,
        [](const dim_info& di) { return di.ydot - 1; },
        [](const dim_info& di) { return di.stride - 1; });
    const std::vector<std::size_t> ones(padding.size(), 1);
    padding.insert(padding.end(), padding.begin(), padding.end());
    auto conv = m.insert_instruction(
        ins,
        make_op("convolution", {{"padding", padding}, {"stride", ones}, {"dilation", ones}}),
        x,
        cw);

    std::vector<std::size_t> merge_dims = {inp_lens[0], w_dims[0]};
    std::transform(dims.begin(),
                   dims.end(),
                   std::back_inserter(merge_dims),
                   [](const dim_info& di) { return di.htilda * di.stride; });
    return m.insert_instruction(ins, make_op("reshape", {{"dims", merge_dims}}), conv);
}

// The residues of a backward convolution, placed on the output grid before the padding is
// cropped
instruction_ref reassemble_residues(module& m,
                                    instruction_ref ins,
                                    const std::vector<dim_info>& dims,
                                    int group,
                                    bool enable_nd_convolution)
{
    auto inp                 = ins->inputs().at(0);
    auto w                   = ins->inputs().at(1);
    const auto& inp_lens     = inp->get_shape().lens();
    const std::size_t k_chan = inp_lens.at(1); // inp channels == forward output channels
    const std::size_t c_pg   = w->get_shape().lens().at(1); // output channels per group

    // With unit stride there is no upsampling, so the single forward convolution is the result.
    // With no dilation every output position is covered exactly once, so the residues reassemble
    // with a pure interleave (concat + reshape/transpose, no pad/add kernels); the `stride <= y`
    // check guarantees no residue is empty, keeping the interleave grid full.
    const bool no_upsample =
        std::all_of(dims.begin(), dims.end(), [](const dim_info& di) { return di.stride == 1; });
    const bool interleave = std::all_of(dims.begin(), dims.end(), [](const dim_info& di) {
        return di.dilation == 1 and di.stride <= di.y;
    });
    // One convolution gives every residue the same number of taps, so the filter must be a
    // multiple of the stride: otherwise it would compute taps of zeros that the residue
    // convolutions leave out.
    const bool even_taps = std::all_of(
        dims.begin(), dims.end(), [](const dim_info& di) { return di.y % di.stride == 0; });
    // A convolution of one input channel is channelwise, which targets handle on its own. The
    // weights must be constant so the reordered filter is folded.
    if(enable_nd_convolution and interleave and even_taps and not no_upsample and group == 1 and
       k_chan != 1 and w->can_eval())
        return residue_convolution(m, ins, inp, w, dims);

    // The residues form a mixed-radix grid, one axis per spatial dim with radix ytilda. A shape
    // over those lengths maps each linear residue index `r` back to its per-dim `itilda`.
    std::vector<std::size_t> ytilda_lens(dims.size());
    std::transform(dims.begin(), dims.end(), ytilda_lens.begin(), [](const dim_info& di) {
        return di.ytilda;
    });
    const shape residue_grid{shape::uint32_type, ytilda_lens};

    const auto res = build_partials(m, ins, inp, w, residue_grid, dims, group, k_chan, c_pg);

    // Residue 0 (itilda all zero) is never empty since every filter axis has length >= 1, so the
    // reassembly always has at least one partial to start from.
    assert(not res.partials.empty());

    if(no_upsample)
        return res.partials.front();
    if(interleave)
        return reassemble_interleave(m, ins, res.partials, dims, inp_lens[0], c_pg * group);
    return reassemble_general(m, ins, res, dims);
}

// Decompose a single convolution_backwards instruction into the v4r1 forward-conv subgraph.
void rewrite_conv_backwards(module& m, instruction_ref ins, bool enable_nd_convolution)
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
    if(group <= 0 or inp_lens.at(1) % group != 0)
        return;

    // stride/dilation/padding and the tensor spatial axes all share the rank nsp; the zips in
    // compute_dims and the per-dim indexing below assume it.
    assert(dilation.size() == nsp and padding.size() >= nsp and inp_lens.size() == nsp + 2 and
           w_lens.size() == nsp + 2 and out_lens.size() == nsp + 2);

    const auto dims = compute_dims(stride, dilation, inp_lens, w_lens);
    auto acc        = reassemble_residues(m, ins, dims, group, enable_nd_convolution);

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

// A broadcast of a tensor that has one value per channel (dim 1)
MIGRAPHX_PRED_MATCHER(channel_broadcast, instruction_ref ins)
{
    if(not contains({"broadcast", "multibroadcast"}, ins->name()))
        return false;
    if(ins->inputs().size() != 1 or ins->get_shape().dynamic())
        return false;
    const auto& in_lens  = ins->inputs().front()->get_shape().lens();
    const auto& out_lens = ins->get_shape().lens();
    // Output dim of the first input dim
    std::size_t offset = out_lens.size() - in_lens.size();
    if(ins->name() == "broadcast")
        offset = ins->get_operator().to_value().at("axis").to<std::size_t>();
    if(offset > 1 or offset + in_lens.size() < 2)
        return false;
    return in_lens[1 - offset] == out_lens[1] and
           std::count(in_lens.begin(), in_lens.end(), 1) + 1 >= in_lens.size();
}

// The dims of the convolution that computes a pixel shuffle of a convolution's output: a
// reshape that splits the output channels, (N, K * r..., spatial...) -> (N, split...,
// spatial...) with K among the split dims, and a transpose that puts K first and each r after
// a spatial dim
struct pixel_shuffle_dims
{
    // Dims of the reshape
    std::vector<std::size_t> dims;
    // Number of dims the output channels are split into besides K
    std::size_t nsplit = 0;
    // Dims of the transpose, which are those of the new convolution, in terms of the reshape
    std::vector<int64_t> perm;

    // Dim of the reshape that holds the output channels of the new convolution
    int64_t channel() const { return perm[1]; }

    bool is_split(int64_t d) const { return d >= 1 and d < 2 + nsplit and d != channel(); }

    bool is_spatial(int64_t d) const { return d >= 2 + nsplit; }

    // Dim of the original convolution that a dim of the reshape that is not split comes from
    std::size_t conv_dim(int64_t d) const
    {
        if(d == channel())
            return 1;
        return d == 0 ? 0 : d - nsplit;
    }

    // The transpose keeps the batch, takes K from the split dims and puts each of the others
    // after a spatial dim, which a split of more than one element then subdivides
    bool valid() const
    {
        if(perm[0] != 0 or perm[1] < 1 or perm[1] >= 2 + nsplit)
            return false;
        auto positions = range(perm.size());
        return std::all_of(positions.begin(), positions.end(), [&](auto i) {
            if(not is_split(perm[i]))
                return true;
            return dims[perm[i]] > 1 and is_spatial(perm[i - 1]);
        });
    }

    // Lens of the transpose's output with each split dim merged into the spatial dim before it
    std::vector<std::size_t> merged_lens() const
    {
        return std::accumulate(
            perm.begin(), perm.end(), std::vector<std::size_t>{}, [&](auto result, auto d) {
                if(is_split(d))
                    result.back() *= dims[d];
                else
                    result.push_back(dims[d]);
                return result;
            });
    }

    // A value per spatial dim of the new convolution: split(d) for dim d of the reshape when
    // it is a split dim, else conv(i) for spatial dim i of the original convolution
    template <class Split, class Conv>
    std::vector<std::size_t> spatial(Split split, Conv conv) const
    {
        std::vector<std::size_t> result(perm.size() - 2);
        std::transform(perm.begin() + 2, perm.end(), result.begin(), [&](auto d) -> std::size_t {
            if(is_split(d))
                return split(d);
            return conv(conv_dim(d) - 2);
        });
        return result;
    }

    // A reshape adds the unit dims: unlike an unsqueeze it gives them strides that follow
    // the layout of x. Only a transpose that reorders the dims of x is left.
    instruction_ref input(module& m, instruction_ref pos, instruction_ref x) const
    {
        std::vector<int64_t> x_perm;
        transform_if(
            perm.begin(),
            perm.end(),
            std::back_inserter(x_perm),
            [&](auto d) { return not is_split(d); },
            [&](auto d) { return conv_dim(d); });
        std::vector<int64_t> x_dims(perm.size());
        std::transform(perm.begin(), perm.end(), x_dims.begin(), [&](auto d) -> int64_t {
            return is_split(d) ? 1 : x->get_shape().lens()[conv_dim(d)];
        });
        if(not std::is_sorted(x_perm.begin(), x_perm.end()))
            x = m.insert_instruction(pos, make_op("transpose", {{"permutation", x_perm}}), x);
        return m.insert_instruction(pos, make_op("reshape", {{"dims", x_dims}}), x);
    }

    // (K * r..., C, k...) -> (split..., C, k...) -> (K, C, r and k...) in the new dim order.
    // Output i of a split dim reads tap r - 1 - i, so the taps are the outputs reversed.
    instruction_ref weights(module& m, instruction_ref pos, instruction_ref w) const
    {
        const auto& w_lens = w->get_shape().lens();
        std::vector<int64_t> w_dims(dims.begin() + 1, dims.begin() + 2 + nsplit);
        w_dims.insert(w_dims.end(), w_lens.begin() + 1, w_lens.end());
        // Dims 1 to nsplit + 1 of the reshape are dims 0 to nsplit of w_dims, and the input
        // channels follow them where the reshape has its first spatial dim
        std::vector<int64_t> w_perm(perm.size());
        std::transform(perm.begin(), perm.end(), w_perm.begin(), [&](auto d) {
            return is_spatial(d) ? d : d - 1;
        });
        w_perm[0] = channel() - 1;
        w_perm[1] = nsplit + 1;
        std::vector<int64_t> axes;
        auto positions = range(perm.size());
        std::copy_if(positions.begin(), positions.end(), std::back_inserter(axes), [&](auto i) {
            return is_split(perm[i]);
        });
        w = m.insert_instruction(pos, make_op("reshape", {{"dims", w_dims}}), w);
        w = m.insert_instruction(pos, make_op("transpose", {{"permutation", w_perm}}), w);
        return m.insert_instruction(pos, make_op("reverse", {{"axes", axes}}), w);
    }

    // The per-channel bias as (1, K, r or 1...) in the new dim order
    instruction_ref bias(module& m, instruction_ref pos, instruction_ref b) const
    {
        std::vector<int64_t> bias_dims(dims.size(), 1);
        std::copy(dims.begin() + 1, dims.begin() + 2 + nsplit, bias_dims.begin() + 1);
        b = m.insert_instruction(pos, make_op("reshape", {{"dims", bias_dims}}), b);
        return m.insert_instruction(pos, make_op("transpose", {{"permutation", perm}}), b);
    }
};

// A pointwise op of one input whose result is used once
bool is_unary_pointwise(instruction_ref ins)
{
    return ins->inputs().size() == 1 and ins->outputs().size() == 1 and
           ins->get_operator().attributes().contains("pointwise");
}

// Whether the output of ins, through single-use pointwise ops, is reshaped to lens
bool reshaped_to(instruction_ref ins, const std::vector<std::size_t>& lens)
{
    while(ins->outputs().size() == 1)
    {
        ins = ins->outputs().front();
        if(ins->name() == "reshape")
            return ins->get_shape().lens() == lens;
        if(not ins->get_operator().attributes().contains("pointwise"))
            return false;
    }
    return false;
}

// A pixel shuffle splits the output channels of a convolution into {K, r...} and interleaves
// each r with a spatial dim. That is a convolution with a spatial dim for each r where the
// input has one element: with a filter of r taps and a padding of r - 1, output i along the
// dim reads the input through tap r - 1 - i alone. The transpose becomes the order of the
// convolution's dims, so the convolution writes the shuffled output:
//
//   transpose(reshape(convolution(x, w) + bias)) -> convolution(x', w') + bias'
//
// Unary ops before the reshape, where simplify_reshapes moves them, are applied to the new
// convolution as they are.
//
// Only reshapes that a later reshape merges into the spatial dims are taken, which leaves the
// other splits of the output channels alone. The weights must be constant so the reordered
// filter is folded.
struct find_convolution_pixel_shuffle
{
    auto matcher() const
    {
        auto conv =
            match::name("convolution")(match::used_once(), match::arg(1)(match::is_constant()))
                .bind("conv");
        auto bias = match::name("add")(
            match::used_once(), match::either_arg(0, 1)(conv, channel_broadcast().bind("bias")));
        auto unary  = match::make_basic_pred_matcher(&is_unary_pointwise);
        auto output = match::skip(unary)(match::any_of(conv, bias));
        auto reshape =
            match::name("reshape")(match::used_once(), match::arg(0)(output)).bind("reshape");
        return match::name("transpose")(match::arg(0)(reshape));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto conv = r.instructions["conv"];
        if(conv->get_shape().dynamic() or ins->get_shape().dynamic())
            return;
        auto v = conv->get_operator().to_value();
        if(v.at("group").to<std::size_t>() != 1 or v.at("padding_mode").to<int>() != 0)
            return;
        // A convolution of one input channel is channelwise, which targets handle on its own
        if(conv->inputs().front()->get_shape().lens()[1] == 1)
            return;
        const auto& conv_lens = conv->get_shape().lens();
        const auto& dims      = r.instructions["reshape"]->get_shape().lens();
        if(dims.size() <= conv_lens.size())
            return;
        // The reshape must keep the batch and the spatial dims
        const auto nsplit = dims.size() - conv_lens.size();
        if(dims.front() != conv_lens.front() or
           not std::equal(conv_lens.begin() + 2, conv_lens.end(), dims.begin() + 2 + nsplit))
            return;
        pixel_shuffle_dims d{
            dims, nsplit, ins->get_operator().to_value().at("permutation").to_vector<int64_t>()};
        if(not d.valid() or not reshaped_to(ins, d.merged_lens()))
            return;

        const auto nspatial = conv_lens.size() - 2;
        auto padding        = v.at("padding").to_vector<std::size_t>();
        if(padding.size() == nspatial)
            padding.insert(padding.end(), padding.begin(), padding.begin() + nspatial);
        auto x = conv->inputs().front();
        // An asymmetric padding is made an explicit pad of the input (see insert_pad). Doing it
        // here keeps the pad off the dims where the input has one element.
        if(not std::equal(padding.begin(), padding.begin() + nspatial, padding.begin() + nspatial))
        {
            std::vector<int64_t> pads(2 * conv_lens.size(), 0);
            std::copy(padding.begin(), padding.begin() + nspatial, pads.begin() + 2);
            std::copy(
                padding.begin() + nspatial, padding.end(), pads.begin() + conv_lens.size() + 2);
            x = m.insert_instruction(ins, make_op("pad", {{"pads", pads}}), x);
            std::fill(padding.begin(), padding.end(), 0);
        }
        const auto stride   = v.at("stride").to_vector<std::size_t>();
        const auto dilation = v.at("dilation").to_vector<std::size_t>();
        const auto full     = [&](auto i) { return dims[i] - 1; };
        const auto one      = [](auto) { return 1; };
        auto new_padding    = d.spatial(full, [&](auto i) { return padding[i]; });
        auto end_padding    = d.spatial(full, [&](auto i) { return padding[i + nspatial]; });
        new_padding.insert(new_padding.end(), end_padding.begin(), end_padding.end());
        v["padding"]  = new_padding;
        v["stride"]   = d.spatial(one, [&](auto i) { return stride[i]; });
        v["dilation"] = d.spatial(one, [&](auto i) { return dilation[i]; });

        auto result = m.insert_instruction(ins,
                                           make_op("convolution", v),
                                           d.input(m, ins, x),
                                           d.weights(m, ins, conv->inputs().back()));
        if(contains(r.instructions, "bias"))
        {
            auto bias = m.insert_instruction(
                ins,
                make_op("multibroadcast", {{"out_lens", result->get_shape().lens()}}),
                d.bias(m, ins, r.instructions["bias"]->inputs().front()));
            result = m.insert_instruction(ins, make_op("add"), result, bias);
        }
        const auto unary_ops =
            get_input_ops_if(r.instructions["reshape"]->inputs().front(), &is_unary_pointwise)
                .second;
        result = std::accumulate(
            unary_ops.begin(), unary_ops.end(), result, [&](auto input, const auto& op) {
                return m.insert_instruction(ins, op, input);
            });
        m.replace_instruction(ins, result);
    }
};
} // namespace

void rewrite_convolution::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution_backwards")
            continue;
        rewrite_conv_backwards(m, ins, enable_nd_convolution);
    }
    if(enable_nd_convolution)
        match::find_matches(m, find_convolution_pixel_shuffle{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
