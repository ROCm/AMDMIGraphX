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
#ifndef MIGRAPHX_GUARD_OPERATORS_POOLING_HPP
#define MIGRAPHX_GUARD_OPERATORS_POOLING_HPP

#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/dyn_output.hpp>
#include <cmath>
#include <utility>

namespace migraphx {

inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pooling
{
    pooling_mode mode = {pooling_mode::average};

    // Padding along each spatial input dimension
    // Can be ndim or 2*ndim values where ndim is size of lengths
    // ndim values means pad the same before and after each dimension
    // 2*ndim values contains n pre and then n post padding values
    std::vector<std::size_t> padding = {0, 0};

    // Size of stride to take from one placement of the pooling kernel to the next.
    // This is distinct from the strides used by the shape class.  Must be the same
    // ndim as lengths.
    std::vector<std::size_t> stride = {1, 1};

    // Spatial dimensions of the pooling kernel or window,
    // 2 smaller than the input tensor rank (NCHW layout)
    std::vector<std::size_t> lengths = {1, 1};

    // Spacing between the elements of the pooling kernel. Must be the same ndim as lengths.
    std::vector<std::size_t> dilations = {1, 1};

    // ceiling mode is a flag affecting output size
    // or equivalently, placements of the pooling kernel.
    // When true, round the size upwards, possibly
    // including partial placements where the kernel extends beyond the edge
    // of input and even padding.  When false, round down so that all
    // kernel placements fit but some input values may be dropped.
    bool ceil_mode = false;
    int lp_order   = 2;

    // Global pooling with dynamic shape input
    bool dyn_global = false;

    // an attribute of the Onnx pooling operator, not currently enabled here because MIOpen can't
    // support it. We currently implement padding for average pooling by inserting a Padding
    // operator during Onnx parsing. But to support dynamic shape inputs and count_include_pad
    // together, it would be necessary to do this calculation at runtime in MIOpen.
    bool count_include_pad = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"),
                    f(self.dilations, "dilations"),
                    f(self.ceil_mode, "ceil_mode"),
                    f(self.lp_order, "lp_order"),
                    f(self.dyn_global, "dyn_global"));
    }

    std::string name() const { return "pooling"; }

    void check_attribute_size() const
    {
        if(dyn_global)
            return;
        if((padding.size() != stride.size() and (padding.size()) != stride.size() * 2) or
           stride.size() != lengths.size() or dilations.size() != lengths.size())
        {
            MIGRAPHX_THROW("POOLING: inconsistent attribute sizes");
        }

        const auto is_zero = [](auto el) { return el == 0; };
        if(std::any_of(lengths.begin(), lengths.end(), is_zero) or
           std::any_of(stride.begin(), stride.end(), is_zero) or
           std::any_of(dilations.begin(), dilations.end(), is_zero))
        {
            MIGRAPHX_THROW("POOLING: size 0 pooling kernel or stride");
        }

        // TODO:  update lowering to run the reference
        // code when OneDNN can't execute pooling for a CPU

        // OneDNN has a limitation on padding size for pooling.  see
        // https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html#doxid-dev-guide-convolution

        // padding = {2}; stride = {1}; lengths = {3} succeeds in oneDNN but
        // padding = {2}; stride = {1}; lengths = {2} fails.
        // Also, the referenced documentation contains a max. dimension size of 14 for the kernel
        // ("weights tensor") that MIGraphX doesn't enforce.
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    std::size_t dilate_dim(std::size_t dim, std::size_t dilation) const
    {
        return dilation * (dim - 1) + 1;
    }

    std::vector<std::size_t> calc_spatial_dim_out(const std::vector<std::size_t>& input_lens,
                                                  std::size_t kdims) const
    {
        std::vector<std::size_t> output_lens{};
        for(size_t i = 0; i < kdims; ++i)
        {
            std::size_t padding_factor = 2 * padding[i];
            if(padding.size() == 2 * kdims)
                padding_factor = padding[i] + padding[i + kdims];
            std::size_t dilated_length = dilate_dim(lengths[i], dilations[i]);
            assert(input_lens[i + 2] + padding_factor >= dilated_length);
            std::size_t dim_size = input_lens[i + 2] + padding_factor - dilated_length;
            std::size_t len =
                (ceil_mode)
                    ? dim_size / stride[i] +
                          static_cast<std::size_t>((dim_size % stride[i] != 0)) // ceil uint divide
                    : dim_size / stride[i];                                     // floor divide
            output_lens.push_back(len + 1);
        }
        return output_lens;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        check_attribute_size();

        const shape& input = inputs.at(0);
        auto padding_size  = padding.size();
        size_t kdims       = input.ndim() - 2;
        if(input.ndim() < 3)
        {
            MIGRAPHX_THROW("POOLING: input must have 3 or more dimensions and be nonempty");
        }
        if(input.ndim() * 2 != padding_size + 4 and input.ndim() != padding_size + 2)
        {
            MIGRAPHX_THROW("POOLING: input and attribute size mismatch!");
        }

        if(input.dynamic())
        {
            auto input_dyn_dims = input.dyn_dims();
            std::vector<shape::dynamic_dimension> output_dyn_dims(input_dyn_dims.begin(),
                                                                  input_dyn_dims.begin() + 2);
            if(dyn_global)
            {
                for(size_t i = 0; i < kdims; ++i)
                {
                    output_dyn_dims.push_back(shape::dynamic_dimension{1, 1});
                }
                return {input.type(), output_dyn_dims};
            }
            else
            {
                // does not compute optimals
                auto min_spatial_dims = calc_spatial_dim_out(input.min_lens(), kdims);
                auto max_spatial_dims = calc_spatial_dim_out(input.max_lens(), kdims);
                for(size_t i = 0; i < kdims; ++i)
                {
                    output_dyn_dims.push_back(
                        shape::dynamic_dimension{min_spatial_dims[i], max_spatial_dims[i], {}});
                }
                return {input.type(), output_dyn_dims};
            }
        }
        else
        {
            auto input_lens = input.lens();

            std::vector<std::size_t> output_lens(input_lens.begin(), input_lens.begin() + 2);
            // Used for when normalize_compute_shape() is called again at model eval time
            // for an originally dynamic shape. Kernel shape is not used with dyn_global.
            if(dyn_global)
            {
                for(size_t i = 0; i < kdims; ++i)
                {
                    output_lens.push_back(1);
                }
                return {input.type(), output_lens};
            }
            else
            {
                auto output_spatial_lens = calc_spatial_dim_out(input_lens, kdims);
                output_lens.insert(
                    output_lens.end(), output_spatial_lens.begin(), output_spatial_lens.end());
                return inputs[0].with_lens(output_lens);
            }
        }
    }

    struct lpnorm_pool
    {
        int p = 0;

        lpnorm_pool() = delete;

        explicit lpnorm_pool(int x) : p{x} {};

        template <class T>
        double init() const
        {
            return 0.0;
        }

        double operator()(double x, double y) const { return x + std::pow(std::abs(y), p); }

        double final(double x, std::size_t) const { return (p == 0) ? 1 : std::pow(x, 1. / p); }
    };

    struct avg_pool
    {
        template <class T>
        double init() const
        {
            return 0.0;
        }

        double operator()(double x, double y) const { return x + y; }

        double final(double x, std::size_t y) const { return (y == 0) ? 0.0 : (x / y); }
    };

    struct max_pool
    {
        template <class T>
        T init() const
        {
            return std::numeric_limits<T>::lowest();
        }

        double operator()(double x, double y) const { return std::max(x, y); }

        double final(double x, std::size_t) const { return (x); }
    };

    template <class Type, class Out, class In, class Op>
    void calc_pooling(const shape& output_shape,
                      Out& output,
                      const In& input,
                      const std::vector<std::size_t>& kernel_dims,
                      Op op) const
    {
        auto in_s    = input.get_shape();
        auto in_lens = in_s.lens();

        // For each element of output; i.e., for each placement of pooling kernel...
        par_for(output_shape.elements(), [&](auto i) {
            auto idx_o = output_shape.multi(i);
            auto n_dim = idx_o.size();
            // starting offset of the pooling window
            std::vector<int> win_start;
            std::vector<std::size_t> win_size;

            // For each spatial dimension, find starting and ending index of pooling kernel
            for(std::size_t dim = 2; dim < n_dim; ++dim)
            {
                auto d_2 = dim - 2;
                int start =
                    static_cast<int>(idx_o[dim] * stride[d_2]) - static_cast<int>(padding[d_2]);
                int end;
                std::size_t dilated_kernel_dim = dilate_dim(kernel_dims[d_2], dilations[d_2]);
                // NOLINT
                if(count_include_pad and ceil_mode and (mode != pooling_mode::max))
                {
                    // TODO: this block can't execute until we enable count_include_pad
                    // Even when using padding, if in ceil_mode a window
                    // could extend beyond the end of both input and
                    // padding.  Clip out-of-bounds indexes but not padding.

                    // Check if this kernel extends beyond the padding at end of dimension
                    end = std::min(start + dilated_kernel_dim,
                                   in_lens[dim] + static_cast<int>(padding[d_2]));
                }
                else
                {
                    // In non-ceiling mode, when
                    // count_include_pad is false, or for max pooling, clip off padding.
                    end   = std::min(start + dilated_kernel_dim, in_lens[dim]);
                    start = std::max(start, 0);
                }
                win_start.push_back(start);
                if(end < start)
                {
                    // This error can be caused by misc. bad input combinations
                    MIGRAPHX_THROW("POOLING:  invalid attributes");
                }
                win_size.push_back(end - start);
            }

            shape win_shape{output_shape.type(), win_size};
            auto pool_size = std::accumulate(
                kernel_dims.cbegin(), kernel_dims.cend(), 1, std::multiplies<std::size_t>());
            double output_val = op.template init<Type>();

            // for each element in the window...
            shape_for_each(win_shape, [&](auto idx_w) {
                // Skip elements that belong to the dilated area
                for(int axis = 0; axis < idx_w.size(); ++axis)
                {
                    if(idx_w[axis] % dilations[axis])
                        return;
                }

                // the coordinates of this element
                auto idx = idx_o;

                // Add the kernel location idx_w and the offset win_start, for each dimension.
                // Negative results are cast to very large unsigned integers.
                std::transform(idx_w.begin(),
                               idx_w.end(),
                               win_start.begin(),
                               idx.begin() + 2,
                               [](auto ii, auto jj) { return ii + jj; });
                // Check if any of coordinates are out of input tensor's range
                if(std::mismatch(idx.begin() + 2,
                                 idx.end(),
                                 in_lens.begin() + 2,
                                 in_lens.end(),
                                 std::less<>{}) == std::make_pair(idx.end(), in_lens.end()))
                {
                    output_val = op(output_val, input[in_s.index(idx)]);
                }
                else
                {
                    // this is a padding element.  Padding locations
                    // don't contribute to average or max pooling total but can play in
                    // lpnorm pooling.
                    output_val = op(output_val, 0);
                }
            });
            output[i] = Type(op.final(output_val, pool_size));
        });
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        auto input_lens = args[0].get_shape().lens();
        std::vector<std::size_t> kernel_dims;
        if(dyn_global)
        {
            kernel_dims.insert(kernel_dims.end(), input_lens.begin() + 2, input_lens.end());
        }
        else
        {
            kernel_dims = this->lengths;
        }
        visit_all(result, args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            switch(mode)
            {
            case migraphx::op::pooling_mode::average:
                calc_pooling<type>(dyn_out.computed_shape, output, input, kernel_dims, avg_pool{});
                break;
            case migraphx::op::pooling_mode::max:
                calc_pooling<type>(dyn_out.computed_shape, output, input, kernel_dims, max_pool{});
                break;
            case migraphx::op::pooling_mode::lpnorm:
                calc_pooling<type>(
                    dyn_out.computed_shape, output, input, kernel_dims, lpnorm_pool{lp_order});
                break;
            }
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
