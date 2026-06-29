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
#include <migraphx/convolution.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/shape_for_each.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#if MIGRAPHX_USE_EIGEN
#include <Eigen/Core>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

template <class Output, class Input>
[[maybe_unused]] void convolution_naive(Output output,
                                        Input input,
                                        Input weights,
                                        const std::vector<std::size_t>& padding,
                                        const std::vector<std::size_t>& stride,
                                        const std::vector<std::size_t>& dilation,
                                        int group)
{
    auto output_shape = output.get_shape();
    auto in_lens      = input.get_shape().lens();

    auto wei_lens = weights.get_shape().lens();
    auto wei_n    = wei_lens[0];
    auto wei_c    = wei_lens[1];
    std::vector<std::size_t> win_size(wei_lens.begin() + 1, wei_lens.end());

    par_for(output_shape.elements(), [&](auto i) {
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

        shape win_shape{output_shape.type(), win_size};

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

        output[i] = acc;
    });
}

#if MIGRAPHX_USE_EIGEN

// An N-d convolution problem and the im2col build/scatter it drives.
struct conv_problem
{
    shape kernel_shape;                  // spatial kernel extents
    shape out_shape;                     // spatial output extents
    std::vector<std::size_t> stride;     // spatial stride
    std::vector<std::size_t> dilation;   // spatial dilation
    std::vector<std::size_t> padding;    // begin padding per spatial dim
    std::vector<std::size_t> in_spatial; // spatial input extents
    std::size_t cpg          = 0;        // input channels per group
    std::size_t kpg          = 0;        // output channels per group
    std::size_t out_channels = 0;        // total output channels
    std::size_t col_count    = 0;        // batch * output positions (im2col columns)

    std::size_t kernel_elems() const { return kernel_shape.elements(); }
    std::size_t out_elems() const { return out_shape.elements(); }
    std::size_t p_dim() const { return cpg * kernel_elems(); } // im2col rows per group

    // Map an output position `os` and kernel tap `ks` to input spatial coordinates, written into
    // `idx` (after the batch/channel slots). Returns false when the tap lands in the padding
    // region.
    bool conv_input_coords(const std::vector<std::size_t>& os,
                           const std::vector<std::size_t>& ks,
                           std::vector<std::ptrdiff_t>& idx) const
    {
        for(std::size_t d = 0; d < stride.size(); ++d)
        {
            const std::ptrdiff_t pos = std::ptrdiff_t(os[d] * stride[d] + ks[d] * dilation[d]) -
                                       std::ptrdiff_t(padding[d]);
            if(pos < 0 or pos >= std::ptrdiff_t(in_spatial[d]))
                return false;
            idx[d + 2] = pos;
        }
        return true;
    }

    // Row-major layout of the im2col matrix, indexed as [channel, tap, column].
    shape col_layout() const { return shape{out_shape.type(), {cpg, kernel_elems(), col_count}}; }

    // Kernel tap coordinates indexed by their row-major tap number; the same for every group.
    std::vector<std::vector<std::size_t>> kernel_taps() const
    {
        std::vector<std::vector<std::size_t>> taps;
        taps.reserve(kernel_elems());
        auto tap_ids = range(kernel_elems());
        std::transform(tap_ids.begin(), tap_ids.end(), std::back_inserter(taps), [&](auto kk) {
            return kernel_shape.multi(kk);
        });
        return taps;
    }

    // Build the im2col matrix for one group as a [Cpg*kernel, batch*output] row-major buffer; each
    // column gathers the receptive field of one (batch, output position) pair. `taps`/`layout` are
    // group-independent and passed in so the caller computes them once across all groups.
    template <class Input>
    void fill_im2col(std::vector<double>& col_buf,
                     Input input,
                     std::size_t g,
                     const std::vector<std::vector<std::size_t>>& taps,
                     const shape& layout) const
    {
        assert(layout.elements() == col_buf.size());
        const std::size_t kdims     = stride.size();
        const std::size_t out_count = out_elems();
        par_for(col_count, [&](auto col) {
            const auto os = out_shape.multi(col % out_count);
            std::vector<std::ptrdiff_t> idx(kdims + 2);
            idx[0] = col / out_count; // batch
            for(std::size_t kk = 0; kk < taps.size(); ++kk)
            {
                // The input position is the same for every channel, so resolve it once per tap.
                if(not conv_input_coords(os, taps[kk], idx))
                    continue;
                for(std::size_t ci = 0; ci < cpg; ++ci)
                {
                    idx[1]                               = g * cpg + ci;
                    col_buf[layout.index({ci, kk, col})] = input(idx.begin(), idx.end());
                }
            }
        });
    }

    // Scatter one group's gemm result [Kpg, batch*output] into the NCHW output buffer.
    template <class Matrix>
    void scatter_conv_group(std::vector<double>& out_buf, const Matrix& c_mat, std::size_t g) const
    {
        const std::size_t out_count = out_elems();
        const std::size_t batch     = col_count / out_count;
        // The gemm result c_mat is a [Kpg, batch, spatial] row-major matrix, so its flat index `i`
        // decodes to those three coordinates...
        const shape result_layout{out_shape.type(), {kpg, batch, out_count}};
        // ...which are scattered into the collapsed output layout [batch, channel, spatial].
        const shape out_layout{out_shape.type(), {batch, out_channels, out_count}};
        assert(result_layout.elements() == static_cast<std::size_t>(c_mat.size()));
        assert(out_layout.elements() == out_buf.size());
        par_for(result_layout.elements(), [&](auto i) {
            const auto m         = result_layout.multi<3>(i); // {filter-in-group, batch, spatial}
            const std::size_t oc = g * kpg + m[0];
            out_buf[out_layout.index({m[1], oc, m[2]})] = c_mat.data()[i];
        });
    }
};

// Reference convolution for any number of spatial dims: build the im2col matrix for each group and
// multiply it by the reshaped weights with an Eigen gemm.
template <class Output, class Input>
void convolution_eigen(Output output,
                       Input input,
                       Input weights,
                       const std::vector<std::size_t>& padding,
                       const std::vector<std::size_t>& stride,
                       const std::vector<std::size_t>& dilation,
                       int group)
{
    using row_major = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    const auto& in_lens     = input.get_shape().lens();
    const auto& wei_lens    = weights.get_shape().lens();
    const auto& out_lens    = output.get_shape().lens();
    const auto type         = output.get_shape().type();
    const std::size_t kdims = stride.size();
    assert(in_lens.size() == kdims + 2 and wei_lens.size() == kdims + 2 and
           out_lens.size() == kdims + 2);

    conv_problem prob;
    prob.kernel_shape = shape{type, std::vector<std::size_t>(wei_lens.begin() + 2, wei_lens.end())};
    prob.out_shape    = shape{type, std::vector<std::size_t>(out_lens.begin() + 2, out_lens.end())};
    prob.stride       = stride;
    prob.dilation     = dilation;
    // Only the begin padding is used (matching the naive impl); drop any trailing end padding.
    prob.padding             = std::vector<std::size_t>(padding.begin(), padding.begin() + kdims);
    prob.in_spatial          = std::vector<std::size_t>(in_lens.begin() + 2, in_lens.end());
    prob.cpg                 = wei_lens[1];
    prob.out_channels        = wei_lens[0];
    const std::size_t groups = group;
    prob.kpg                 = prob.out_channels / groups;
    prob.col_count           = in_lens[0] * prob.out_elems();
    // Groups partition the channels evenly; relied on by the per-group weight/output blocks.
    assert(group > 0 and prob.out_channels % groups == 0 and in_lens[1] == prob.cpg * groups);

    // Weights as a contiguous [Cout, Cpg*kernel] row-major buffer; each group's filters form a
    // contiguous [Kpg, p_dim] block that is used directly as the gemm's left operand.
    std::vector<double> wei_buf(weights.begin(), weights.end());
    std::vector<double> out_buf(output.get_shape().elements());

    // The im2col tap table and column layout are the same for every group, so build them once.
    const auto taps         = prob.kernel_taps();
    const shape col_layout  = prob.col_layout();
    const std::size_t p_dim = prob.p_dim();
    std::vector<double> col_buf(p_dim * prob.col_count);
    for(std::size_t g = 0; g < groups; ++g)
    {
        std::fill(col_buf.begin(), col_buf.end(), 0.0);
        prob.fill_im2col(col_buf, input, g, taps, col_layout);

        // [Kpg, col_count] = [Kpg, p_dim] * [p_dim, col_count]
        Eigen::Map<row_major> a_mat(wei_buf.data() + g * prob.kpg * p_dim, prob.kpg, p_dim);
        Eigen::Map<row_major> b_mat(col_buf.data(), p_dim, prob.col_count);
        row_major c_mat = a_mat * b_mat;

        prob.scatter_conv_group(out_buf, c_mat, g);
    }

    std::copy(out_buf.begin(), out_buf.end(), output.begin());
}

#endif // MIGRAPHX_USE_EIGEN

} // namespace

void convolution(const argument& result,
                 const argument& x,
                 const argument& w,
                 const std::vector<std::size_t>& padding,
                 const std::vector<std::size_t>& stride,
                 const std::vector<std::size_t>& dilation,
                 int group)
{
    result.visit([&](auto output) {
        get_all<double>(x, w)([&](auto input, auto weights) {
#if MIGRAPHX_USE_EIGEN
            convolution_eigen(output, input, weights, padding, stride, dilation, group);
#else
            convolution_naive(output, input, weights, padding, stride, dilation, group);
#endif
        });
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
