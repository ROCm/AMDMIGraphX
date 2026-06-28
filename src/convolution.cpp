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
#include <migraphx/shape.hpp>
#include <migraphx/shape_for_each.hpp>
#include <algorithm>
#include <cstddef>
#include <vector>

#if MIGRAPHX_USE_EIGEN
#include <unsupported/Eigen/CXX11/Tensor>
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

// The begin/end padding for spatial dimension `d`. MIGraphX stores padding either
// as one value per spatial dim (symmetric) or as all begins followed by all ends.
std::pair<std::ptrdiff_t, std::ptrdiff_t> spatial_padding(const std::vector<std::size_t>& padding,
                                                          std::size_t kdims,
                                                          std::size_t d)
{
    std::ptrdiff_t begin = padding[d];
    std::ptrdiff_t end   = padding.size() == 2 * kdims ? padding[kdims + d] : padding[d];
    return {begin, end};
}

// 2D convolution: lower to im2col + matmul with Eigen's `extract_image_patches` and `contract`.
// A MIGraphX NCHW row-major buffer aliases a column-major Eigen tensor with dims [W, H, C, N],
// which is shuffled to the [depth, rows, cols, batch] order that `extract_image_patches` expects.
template <class Output, class Input>
void convolution_eigen_patches(Output output,
                               Input input,
                               Input weights,
                               const std::vector<std::size_t>& padding,
                               const std::vector<std::size_t>& stride,
                               const std::vector<std::size_t>& dilation,
                               int group)
{
    using index   = Eigen::Index;
    using tensor4 = Eigen::Tensor<double, 4>;
    using tensor5 = Eigen::Tensor<double, 5>;
    using map4    = Eigen::TensorMap<tensor4>;

    const auto& in_lens  = input.get_shape().lens();
    const auto& wei_lens = weights.get_shape().lens();
    const auto& out_lens = output.get_shape().lens();

    const index n   = in_lens[0];
    const index c   = in_lens[1];
    const index ih  = in_lens[2];
    const index iw  = in_lens[3];
    const index k   = wei_lens[0]; // total output channels
    const index cpg = wei_lens[1]; // input channels per group
    const index kh  = wei_lens[2];
    const index kw  = wei_lens[3];
    const index oh  = out_lens[2];
    const index ow  = out_lens[3];
    const index kpg = k / group; // output channels per group

    const index sh = stride[0];
    const index sw = stride[1];
    const index dh = dilation[0];
    const index dw = dilation[1];

    const auto pad_h = spatial_padding(padding, 2, 0);
    const auto pad_w = spatial_padding(padding, 2, 1);

    std::vector<double> in_buf(input.get_shape().elements());
    std::copy(input.begin(), input.end(), in_buf.begin());
    std::vector<double> wei_buf(weights.get_shape().elements());
    std::copy(weights.begin(), weights.end(), wei_buf.begin());
    std::vector<double> out_buf(output.get_shape().elements(), 0.0);

    // Column-major aliases of the row-major NCHW/KCHW/NCHW buffers.
    map4 in_map(in_buf.data(), iw, ih, c, n);
    map4 wei_map(wei_buf.data(), kw, kh, cpg, k);
    map4 out_map(out_buf.data(), ow, oh, k, n);

    const Eigen::array<index, 4> to_depth_major{2, 1, 0, 3}; // [W,H,*,N] -> [*,H,W,N]
    const Eigen::array<index, 4> to_kernel{3, 2, 1, 0};      // [KW,KH,Cpg,Kpg] -> [Kpg,Cpg,KH,KW]
    const Eigen::array<index, 2> patch_2d{cpg * kh * kw, oh * ow * n};
    const Eigen::array<index, 2> kernel_2d{kpg, cpg * kh * kw};
    const Eigen::array<index, 4> out_4d{kpg, oh, ow, n};
    const Eigen::array<Eigen::IndexPair<index>, 1> contract_dims{Eigen::IndexPair<index>{1, 0}};

    for(index g = 0; g < group; ++g)
    {
        // [Cpg, H, W, N]
        tensor4 in_chw =
            in_map.slice(Eigen::array<index, 4>{0, 0, g * cpg, 0}, Eigen::array<index, 4>{iw, ih, cpg, n})
                .shuffle(to_depth_major);

        // [Cpg, KH, KW, OH*OW, N]
        tensor5 patches = in_chw.extract_image_patches(
            kh, kw, sh, sw, dh, dw, 1, 1, pad_h.first, pad_h.second, pad_w.first, pad_w.second, 0.0);

        // [Kpg, Cpg, KH, KW]
        tensor4 wei_g = wei_map
                            .slice(Eigen::array<index, 4>{0, 0, 0, g * kpg},
                                   Eigen::array<index, 4>{kw, kh, cpg, kpg})
                            .shuffle(to_kernel);

        // [Kpg, OH*OW*N] = [Kpg, Cpg*KH*KW] * [Cpg*KH*KW, OH*OW*N]
        Eigen::Tensor<double, 2> result =
            wei_g.reshape(kernel_2d).contract(patches.reshape(patch_2d), contract_dims);

        // Scatter back into the group's output channels (NCHW alias [OW,OH,Cout,N]).
        out_map.slice(Eigen::array<index, 4>{0, 0, g * kpg, 0},
                      Eigen::array<index, 4>{ow, oh, kpg, n}) =
            result.reshape(out_4d).shuffle(to_depth_major);
    }

    std::copy(out_buf.begin(), out_buf.end(), output.begin());
}

// Static description of an N-d convolution, shared by the im2col helpers below.
struct conv_problem
{
    shape kernel_shape;                  // spatial kernel extents
    shape out_shape;                     // spatial output extents
    std::vector<std::size_t> stride;     //
    std::vector<std::size_t> dilation;   //
    std::vector<std::size_t> padding;    // begin padding per spatial dim
    std::vector<std::size_t> in_spatial; // spatial input extents
    std::size_t cpg          = 0;        // input channels per group
    std::size_t kpg          = 0;        // output channels per group
    std::size_t out_channels = 0;        // total output channels
    std::size_t col_count    = 0;        // batch * output positions (im2col columns)

    std::size_t kernel_elems() const { return kernel_shape.elements(); }
    std::size_t out_elems() const { return out_shape.elements(); }
    std::size_t p_dim() const { return cpg * kernel_elems(); } // im2col rows per group
};

// Map an output position `os` and kernel tap `ks` to input spatial coordinates, written into
// `idx` (after the batch/channel slots). Returns false when the tap lands in the padding region.
bool conv_input_coords(const conv_problem& prob,
                       const std::vector<std::size_t>& os,
                       const std::vector<std::size_t>& ks,
                       std::vector<std::ptrdiff_t>& idx)
{
    for(std::size_t d = 0; d < prob.stride.size(); ++d)
    {
        const std::ptrdiff_t pos = std::ptrdiff_t(os[d]) * prob.stride[d] +
                                   std::ptrdiff_t(ks[d]) * prob.dilation[d] -
                                   std::ptrdiff_t(prob.padding[d]);
        if(pos < 0 or pos >= std::ptrdiff_t(prob.in_spatial[d]))
            return false;
        idx[d + 2] = pos;
    }
    return true;
}

// Build the im2col matrix for one group as a [Cpg*kernel, batch*output] row-major buffer; each
// column gathers the receptive field of one (batch, output position) pair.
template <class Input>
void fill_im2col(std::vector<double>& col_buf, Input input, const conv_problem& prob, std::size_t g)
{
    const std::size_t kdims        = prob.stride.size();
    const std::size_t kernel_elems = prob.kernel_elems();
    const std::size_t out_elems    = prob.out_elems();
    par_for(prob.col_count, [&](auto col) {
        const auto os = prob.out_shape.multi(col % out_elems);
        std::vector<std::ptrdiff_t> idx(kdims + 2);
        idx[0] = col / out_elems; // batch
        for(std::size_t ci = 0; ci < prob.cpg; ++ci)
        {
            idx[1] = g * prob.cpg + ci;
            for(std::size_t kk = 0; kk < kernel_elems; ++kk)
            {
                if(conv_input_coords(prob, os, prob.kernel_shape.multi(kk), idx))
                    col_buf[(ci * kernel_elems + kk) * prob.col_count + col] =
                        input(idx.begin(), idx.end());
            }
        }
    });
}

// Scatter one group's gemm result [Kpg, batch*output] into the NCHW output buffer.
template <class Matrix>
void scatter_conv_group(std::vector<double>& out_buf,
                        const Matrix& c_mat,
                        const conv_problem& prob,
                        std::size_t g)
{
    const std::size_t out_elems = prob.out_elems();
    par_for(prob.kpg * prob.col_count, [&](auto i) {
        const std::size_t f   = i / prob.col_count;
        const std::size_t col = i % prob.col_count;
        const std::size_t bn  = col / out_elems;
        const std::size_t oc  = g * prob.kpg + f;
        out_buf[(bn * prob.out_channels + oc) * out_elems + (col % out_elems)] = c_mat(f, col);
    });
}

// N-dimensional convolution: `extract_image_patches` only handles the 2D case, so build the
// im2col matrix explicitly and multiply it by the reshaped weights with an Eigen gemm.
template <class Output, class Input>
void convolution_eigen_im2col(Output output,
                              Input input,
                              Input weights,
                              const std::vector<std::size_t>& padding,
                              const std::vector<std::size_t>& stride,
                              const std::vector<std::size_t>& dilation,
                              int group)
{
    using row_major = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    const auto& in_lens  = input.get_shape().lens();
    const auto& wei_lens = weights.get_shape().lens();
    const auto& out_lens = output.get_shape().lens();
    const auto type      = output.get_shape().type();

    conv_problem prob;
    prob.kernel_shape = shape{type, std::vector<std::size_t>(wei_lens.begin() + 2, wei_lens.end())};
    prob.out_shape    = shape{type, std::vector<std::size_t>(out_lens.begin() + 2, out_lens.end())};
    prob.stride       = stride;
    prob.dilation     = dilation;
    prob.padding      = padding;
    prob.in_spatial   = std::vector<std::size_t>(in_lens.begin() + 2, in_lens.end());
    prob.cpg          = wei_lens[1];
    prob.out_channels = wei_lens[0];
    prob.kpg          = prob.out_channels / group;
    prob.col_count    = in_lens[0] * prob.out_elems();

    // Weights as a contiguous [Cout, Cpg*kernel] row-major buffer; each group's filters form a
    // contiguous [Kpg, p_dim] block that is used directly as the gemm's left operand.
    std::vector<double> wei_buf(weights.get_shape().elements());
    std::copy(weights.begin(), weights.end(), wei_buf.begin());
    std::vector<double> out_buf(output.get_shape().elements(), 0.0);

    const std::size_t p_dim = prob.p_dim();
    for(std::size_t g = 0; g < std::size_t(group); ++g)
    {
        std::vector<double> col_buf(p_dim * prob.col_count, 0.0);
        fill_im2col(col_buf, input, prob, g);

        // [Kpg, col_count] = [Kpg, p_dim] * [p_dim, col_count]
        Eigen::Map<row_major> a_mat(wei_buf.data() + g * prob.kpg * p_dim, prob.kpg, p_dim);
        Eigen::Map<row_major> b_mat(col_buf.data(), p_dim, prob.col_count);
        row_major c_mat = a_mat * b_mat;

        scatter_conv_group(out_buf, c_mat, prob, g);
    }

    std::copy(out_buf.begin(), out_buf.end(), output.begin());
}

template <class Output, class Input>
void convolution_eigen(Output output,
                       Input input,
                       Input weights,
                       const std::vector<std::size_t>& padding,
                       const std::vector<std::size_t>& stride,
                       const std::vector<std::size_t>& dilation,
                       int group)
{
    if(stride.size() == 2)
        convolution_eigen_patches(output, input, weights, padding, stride, dilation, group);
    else
        convolution_eigen_im2col(output, input, weights, padding, stride, dilation, group);
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
