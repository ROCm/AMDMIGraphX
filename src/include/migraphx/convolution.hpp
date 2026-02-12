/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP

#include <migraphx/config.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/tensor_view.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Output, class T, class Padding, class Stride, class Dilation>
void convolution(
    Output output, T input, T weights, Padding padding, Stride stride, Dilation dilation, int group)
{
    auto output_shape = output.get_shape();
    auto in_lens      = input.get_shape().lens();

    auto wei_lens = weights.get_shape().lens();
    auto wei_n    = wei_lens[0];
    auto wei_c    = wei_lens[1];
    std::vector<std::size_t> win_size(wei_lens.begin() + 1, wei_lens.end());

    shape win_shape{output_shape.type(), win_size};

    // Compute the convolution output value for a single output element index
    auto compute = [&](auto i) {
        auto idx_o = output_shape.multi(i);
        auto w     = idx_o[1];
        auto n_dim = idx_o.size();

        std::vector<std::ptrdiff_t> win_start;
        for(std::size_t dim = 2; dim < n_dim; ++dim)
        {
            auto d_2 = dim - 2;
            win_start.push_back(std::ptrdiff_t(idx_o[dim] * stride[d_2]) -
                                std::ptrdiff_t(padding[d_2]));
        }
        const auto group_id = w / (wei_n / group);

        double acc = 0.0;
        shape_for_each(win_shape, [&](const auto& idx_win) {
            auto k           = idx_win[0];
            const auto in_ch = group_id * wei_c + k;
            std::vector<std::ptrdiff_t> idx(idx_o.begin(), idx_o.end());
            idx[1] = in_ch;
            std::vector<std::ptrdiff_t> idx_dil(idx_win.size() - 1);
            std::transform(idx_win.cbegin() + 1,
                           idx_win.cend(),
                           dilation.cbegin(),
                           idx_dil.begin(),
                           [](std::ptrdiff_t ii, std::ptrdiff_t d) { return d * ii; });
            std::transform(idx_dil.begin(),
                           idx_dil.end(),
                           win_start.begin(),
                           idx.begin() + 2,
                           [](std::ptrdiff_t ii, std::ptrdiff_t jj) { return ii + jj; });
            std::vector<std::ptrdiff_t> idx_wei(idx_o.size());
            idx_wei[0] = w;
            std::copy(idx_win.begin(), idx_win.end(), idx_wei.begin() + 1);
            if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
               std::equal(idx.begin(),
                          idx.end(),
                          in_lens.begin(),
                          in_lens.end(),
                          std::less<std::ptrdiff_t>{}))
            {
                acc += input(idx.begin(), idx.end()) * weights(idx_wei.begin(), idx_wei.end());
            }
        });
        return acc;
    };

    // When the input is spatially broadcast (stride-0 spatial dims, e.g. a
    // broadcast bias), every spatial position in a channel produces the same
    // result. Compute only one position per batch*channel, then fill.
    auto& in_strides = input.get_shape().strides();
    bool spatial_bcast =
        in_strides.size() > 2 and
        std::all_of(in_strides.begin() + 2, in_strides.end(), [](auto s) { return s == 0; });
    if(spatial_bcast)
    {
        auto out_lens       = output_shape.lens();
        std::size_t spatial = 1;
        for(std::size_t d = 2; d < out_lens.size(); d++)
            spatial *= out_lens[d];

        auto n_batch_chan = out_lens[0] * out_lens[1];
        std::vector<double> values(n_batch_chan);
        par_for(n_batch_chan, [&](auto bc) {
            values[bc] = compute(bc * spatial); // first spatial pos
        });

        par_for(n_batch_chan, [&](auto bc) {
            auto val  = values[bc];
            auto base = bc * spatial;
            for(std::size_t s = 0; s < spatial; s++)
                output[base + s] = val;
        });
        return;
    }

    par_for(output_shape.elements(), [&](auto i) { output[i] = compute(i); });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
