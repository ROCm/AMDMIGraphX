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
#include <migraphx/op/convolution_backwards.hpp>

#include <algorithm>
#include <functional>
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
    std::size_t lbuf;   // reassembly buffer length before cropping
    std::size_t y;      // filter length
};

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

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
    {
        auto op  = any_cast<op::convolution_backwards>(ins->get_operator());
        auto dy  = ins->inputs().at(0);
        auto w   = ins->inputs().at(1);
        auto dys = dy->get_shape();
        auto ws  = w->get_shape();
        auto os  = ins->get_shape();

        // The decomposition needs concrete spatial/kernel sizes; leave dynamic shapes untouched
        // so the existing (MLIR / reference) path handles them.
        if(dys.dynamic() or ws.dynamic() or os.dynamic())
            continue;

        const std::size_t nsp    = op.stride.size();
        const auto dy_lens       = dys.lens();
        const auto w_lens        = ws.lens();
        const auto out_lens      = os.lens();
        const std::size_t k_chan = dy_lens.at(1); // dy channels == forward output channels
        const std::size_t c_pg   = w_lens.at(1);  // output channels per group
        const int group          = op.group;
        if(group <= 0 or k_chan % static_cast<std::size_t>(group) != 0)
            continue;

        std::vector<dim_info> dims(nsp);
        for(std::size_t d = 0; d < nsp; ++d)
        {
            auto& di             = dims[d];
            di.stride            = op.stride[d];
            di.dilation          = op.dilation[d];
            di.y                 = w_lens[2 + d];
            const std::size_t ho = dy_lens[2 + d];
            const std::size_t g  = std::gcd(di.stride, di.dilation);
            di.ytilda            = di.stride / g;
            di.dd                = di.dilation / g;
            di.ydot              = ceil_div(di.y, di.ytilda);
            di.htilda            = ho + (di.ydot - 1) * di.dd;
            di.lbuf              = di.stride * di.htilda + (di.ytilda - 1) * di.dilation;
        }

        // Helpers that build the reassembly (insertion point is `ins`).
        auto pad_axis = [&](instruction_ref t, std::size_t axis, int64_t before, int64_t after) {
            if(before == 0 and after == 0)
                return t;
            const std::size_t nd = t->get_shape().ndim();
            std::vector<int64_t> pads(2 * nd, 0);
            pads[axis]      = before;
            pads[nd + axis] = after;
            return m.insert_instruction(ins, make_op("pad", {{"pads", pads}}), t);
        };
        // Insert (step-1) zeros after every element along `axis` (spacing the data out by `step`).
        auto zero_stuff = [&](instruction_ref t, std::size_t axis, std::size_t step) {
            if(step <= 1)
                return t;
            auto u = m.insert_instruction(
                ins, make_op("unsqueeze", {{"axes", {static_cast<int64_t>(axis + 1)}}}), t);
            u         = pad_axis(u, axis + 1, 0, static_cast<int64_t>(step - 1));
            auto lens = u->get_shape().lens();
            std::vector<int64_t> rdims(lens.begin(), lens.end());
            rdims[axis] *= static_cast<int64_t>(step);
            rdims.erase(rdims.begin() + axis + 1);
            return m.insert_instruction(ins, make_op("reshape", {{"dims", rdims}}), u);
        };

        const std::size_t num_res = std::accumulate(
            dims.begin(), dims.end(), std::size_t{1}, [](std::size_t a, const dim_info& di) {
                return a * di.ytilda;
            });

        // When there is no dilation every output position is covered exactly once, so the residues
        // reassemble with a pure interleave (concat + reshape/transpose, no pad/add kernels). The
        // `stride <= y` check guarantees no residue is empty, keeping the interleave grid full.
        const bool interleave = std::all_of(dims.begin(), dims.end(), [](const dim_info& di) {
            return di.dilation == 1 and di.stride <= di.y;
        });

        std::vector<instruction_ref> partials;
        std::vector<std::vector<std::size_t>> partial_itilda;
        for(std::size_t r = 0; r < num_res; ++r)
        {
            // Mixed-radix decode of the residue index into itilda per dim.
            std::vector<std::size_t> itilda(nsp);
            std::size_t rem = r;
            for(std::size_t d = nsp; d-- > 0;)
            {
                itilda[d] = rem % dims[d].ytilda;
                rem /= dims[d].ytilda;
            }

            // Number of filter taps kept by this residue along each dim.
            std::vector<std::size_t> ydot_slice(nsp);
            bool empty = false;
            for(std::size_t d = 0; d < nsp; ++d)
            {
                if(itilda[d] >= dims[d].y)
                {
                    empty = true;
                    break;
                }
                ydot_slice[d] = ceil_div(dims[d].y - itilda[d], dims[d].ytilda);
            }
            if(empty)
                continue;

            // --- weight branch (constant-folded later): strided tap-slice of the filter ---
            auto wsl = w;
            for(std::size_t d = 0; d < nsp; ++d)
            {
                if(itilda[d] > 0) // a starts==0 slice over the full axis is a no-op
                    wsl =
                        m.insert_instruction(ins,
                                             make_op("slice",
                                                     {{"axes", {static_cast<int64_t>(2 + d)}},
                                                      {"starts", {static_cast<int64_t>(itilda[d])}},
                                                      {"ends", {static_cast<int64_t>(dims[d].y)}}}),
                                             wsl);
                if(dims[d].ytilda > 1)
                    wsl = m.insert_instruction(
                        ins,
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
                auto split =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", split_dims}}), wsl);

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
            std::vector<std::size_t> conv_pad(2 * nsp, 0);
            std::vector<std::size_t> conv_str(nsp, 1);
            std::vector<std::size_t> conv_dil(nsp);
            for(std::size_t d = 0; d < nsp; ++d)
            {
                conv_pad[d]       = (ydot_slice[d] - 1) * dims[d].dd; // leading
                conv_pad[nsp + d] = (dims[d].ydot - 1) * dims[d].dd;  // trailing -> common htilda
                conv_dil[d]       = dims[d].dd;
            }
            auto partial = m.insert_instruction(ins,
                                                make_op("convolution",
                                                        {{"padding", conv_pad},
                                                         {"stride", conv_str},
                                                         {"dilation", conv_dil},
                                                         {"group", group}}),
                                                dy,
                                                cw);

            partials.push_back(partial);
            partial_itilda.push_back(itilda);
        }

        // With unit stride there is no upsampling, so the single forward convolution is the result.
        const bool no_upsample = std::all_of(
            dims.begin(), dims.end(), [](const dim_info& di) { return di.stride == 1; });

        instruction_ref acc;
        if(no_upsample)
        {
            acc = partials.front();
        }
        else if(interleave)
        {
            // Pixel-shuffle reassembly: stack the residues on a new trailing axis, then
            // reshape/transpose so residue itilda interleaves into spatial position ht*S + itilda.
            const int64_t new_axis = static_cast<int64_t>(2 + nsp);
            std::vector<instruction_ref> stacked;
            stacked.reserve(partials.size());
            std::transform(
                partials.begin(), partials.end(), std::back_inserter(stacked), [&](auto p) {
                    return m.insert_instruction(
                        ins, make_op("unsqueeze", {{"axes", {new_axis}}}), p);
                });
            auto cat = m.insert_instruction(ins, make_op("concat", {{"axis", new_axis}}), stacked);

            const int64_t n_batch = static_cast<int64_t>(dy_lens[0]);
            const int64_t c_total = static_cast<int64_t>(c_pg) * group;
            // [N, C, *Htilda, num_res] -> [N, C, *Htilda, *S]
            std::vector<int64_t> split_dims{n_batch, c_total};
            for(const auto& di : dims)
                split_dims.push_back(static_cast<int64_t>(di.htilda));
            for(const auto& di : dims)
                split_dims.push_back(static_cast<int64_t>(di.stride));
            // For 1-D the concat already has this shape (num_res == stride), so the split is a
            // no-op.
            const auto cat_lens = cat->get_shape().lens();
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
            auto trans = (perm == identity)
                             ? split // 1-D: the interleave permutation is a no-op
                             : m.insert_instruction(
                                   ins, make_op("transpose", {{"permutation", perm}}), split);
            std::vector<int64_t> merge_dims{n_batch, c_total};
            for(const auto& di : dims)
                merge_dims.push_back(static_cast<int64_t>(di.htilda * di.stride));
            acc = m.insert_instruction(ins, make_op("reshape", {{"dims", merge_dims}}), trans);
        }
        else
        {
            // General reassembly (handles dilation>1 / gcd>1): place each residue onto its
            // sub-lattice via zero-stuff by S and shift by itilda*D, then sum (disjoint supports).
            std::vector<instruction_ref> placed;
            placed.reserve(partials.size());
            for(std::size_t i = 0; i < partials.size(); ++i)
            {
                auto partial       = partials[i];
                const auto& itilda = partial_itilda[i];
                for(std::size_t d = 0; d < nsp; ++d)
                {
                    const std::size_t axis = 2 + d;
                    partial                = zero_stuff(partial, axis, dims[d].stride);
                    const int64_t before   = static_cast<int64_t>(itilda[d] * dims[d].dilation);
                    const int64_t after =
                        static_cast<int64_t>((dims[d].ytilda - 1 - itilda[d]) * dims[d].dilation);
                    partial = pad_axis(partial, axis, before, after);
                }
                placed.push_back(partial);
            }
            acc = placed.front();
            for(std::size_t i = 1; i < placed.size(); ++i)
                acc = m.insert_instruction(ins, make_op("add"), acc, placed[i]);
        }

        // Crop the padding region to produce dx.
        for(std::size_t d = 0; d < nsp; ++d)
        {
            const int64_t start = static_cast<int64_t>(op.padding[d]);
            const int64_t end   = start + static_cast<int64_t>(out_lens[2 + d]);
            if(start == 0 and end == static_cast<int64_t>(acc->get_shape().lens()[2 + d]))
                continue; // crop covers the whole axis -> no-op
            acc = m.insert_instruction(ins,
                                       make_op("slice",
                                               {{"axes", {static_cast<int64_t>(2 + d)}},
                                                {"starts", {start}},
                                                {"ends", {end}}}),
                                       acc);
        }
        m.replace_instruction(ins, acc);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
