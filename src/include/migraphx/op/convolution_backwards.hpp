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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_BACKWARDS_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_BACKWARDS_HPP

#include <cmath>
#include <utility>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/dyn_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct convolution_backwards
{
    std::vector<std::size_t> padding  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

    padding_mode_t padding_mode = default_;
    int group                   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.group, "group"));
    }

    std::string name() const { return "convolution_backwards"; }

    void check_attribute_size() const
    {
        if(padding.size() != stride.size() or stride.size() != dilation.size())
        {
            MIGRAPHX_THROW("CONVOLUTION_BACKWARDS: inconsistent attribute sizes");
        }
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2).same_type().same_ndims().min_ndims(3);

        const shape& x_shape = inputs.at(0);
        const shape& w_shape = inputs.at(1);
        if(x_shape.ndim() - 2 != this->kdims())
        {
            MIGRAPHX_THROW("CONVOLUTION_BACKWARDS: input k-dims does not match attribute size");
        }

        if(group < 1)
        {
            MIGRAPHX_THROW("CONVOLUTION_BACKWARDS: group (" + to_string(group) +
                           ") must be positive");
        }

        if(not x_shape.dynamic() and not w_shape.dynamic() and
           x_shape.lens().at(1) != (w_shape.lens().at(0)))
        {
            MIGRAPHX_THROW("CONVOLUTION_BACKWARDS: mismatched channel numbers");
        }

        // compute() walks a group's input channels as one contiguous block of
        // weights_channels / group, so an inexact division leaves part of every group unread.
        if(not w_shape.dynamic() and w_shape.lens().at(0) % group != 0)
        {
            MIGRAPHX_THROW("CONVOLUTION_BACKWARDS: input channels (" +
                           to_string(w_shape.lens().at(0)) + ") is not divisible by group (" +
                           to_string(group) + ")");
        }

        if(x_shape.dynamic() or w_shape.dynamic())
        {
            return dynamic_compute_shape(x_shape, w_shape);
        }
        else
        {
            return static_compute_shape(x_shape, w_shape);
        }
    }

    std::vector<std::size_t> calc_spatial_lens(std::vector<std::size_t> x_lens,
                                               std::vector<std::size_t> w_lens) const
    {
        std::vector<size_t> spatial_lens(x_lens.size() - 2);

        // stride * (input - 1) + output_padding + ((kernel - 1) * dilation + 1) - padding_L -
        // padding_R. This assumes padding_L = padding_R and output_padding handled in parser.
        for(size_t i = 0; i < spatial_lens.size(); i++)
        {
            spatial_lens.at(i) = (std::size_t(std::max<std::ptrdiff_t>(
                1,
                stride[i] * (x_lens[i + 2] - 1) + ((w_lens[i + 2] - 1) * dilation[i] + 1) -
                    2 * padding[i])));
        }
        return spatial_lens;
    }

    shape dynamic_compute_shape(shape x_shape, shape w_shape) const
    {
        std::vector<shape::dynamic_dimension> output_dyn_dims = {};
        output_dyn_dims.push_back(x_shape.to_dynamic().dyn_dims().at(0));
        output_dyn_dims.push_back(w_shape.to_dynamic().dyn_dims().at(1) * group);
        const std::size_t num_spatial_dims = x_shape.ndim() - 2;
        // Does not compute for optimals
        auto min_spatial_dims = calc_spatial_lens(x_shape.min_lens(), w_shape.min_lens());
        auto max_spatial_dims = calc_spatial_lens(x_shape.max_lens(), w_shape.max_lens());
        for(size_t i = 0; i < num_spatial_dims; ++i)
        {
            output_dyn_dims.push_back(
                shape::dynamic_dimension{min_spatial_dims[i], max_spatial_dims[i], {}});
        }
        return shape{x_shape.type(), output_dyn_dims};
    }

    shape static_compute_shape(shape x_shape, shape w_shape) const
    {
        std::vector<size_t> output_lens{x_shape.lens()[0], w_shape.lens()[1] * group};
        auto spatial_lens = calc_spatial_lens(x_shape.lens(), w_shape.lens());
        std::for_each(spatial_lens.begin(), spatial_lens.end(), [&output_lens](auto x) {
            output_lens.push_back(x);
        });
        return x_shape.with_lens(output_lens);
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        const auto num_spatial_dims = this->kdims();
        const shape& out_shape      = dyn_out.computed_shape;
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            using type = typename decltype(output)::value_type;

            const auto& in_lens = input.get_shape().lens();
            const auto& wei     = weights.get_shape().lens();
            const auto wei_c    = wei[1];
            // Channels of the input, and so of the weights' first axis, per group.
            const auto in_per_group = wei[0] / group;

            const shape wei_spatial{out_shape.type(),
                                    std::vector<std::size_t>(wei.begin() + 2, wei.end())};

            par_for(out_shape.elements(), [&](std::size_t i) {
                const auto idx_out  = out_shape.multi(i);
                const auto group_id = idx_out[1] / wei_c;

                std::vector<std::size_t> idx_in(num_spatial_dims + 2);
                std::vector<std::size_t> idx_wei(num_spatial_dims + 2);
                idx_in[0]  = idx_out[0];
                idx_wei[1] = idx_out[1] % wei_c;

                // Sum in double rather than in the output type: a narrow type (fp16/bf16/fp8)
                // cannot hold the running sum without rounding every term, which loses several
                // ULPs over the channel/kernel extent. migraphx::convolution states the same
                // contract for the forward direction.
                double acc = 0.0;
                shape_for_each(wei_spatial, [&](const auto& idx_k) {
                    // The forward direction sends input position q to q * stride - padding +
                    // k * dilation, so invert that: this weight tap feeds this output position
                    // only when the inverse lands on an input position that exists.
                    for(std::size_t n = 0; n < num_spatial_dims; n++)
                    {
                        const auto pos = std::ptrdiff_t(idx_out[n + 2]) +
                                         std::ptrdiff_t(padding[n]) -
                                         std::ptrdiff_t(idx_k[n] * dilation[n]);
                        if(pos < 0 or pos % std::ptrdiff_t(stride[n]) != 0)
                            return;
                        const auto q = std::size_t(pos) / stride[n];
                        if(q >= in_lens[n + 2])
                            return;
                        idx_in[n + 2]  = q;
                        idx_wei[n + 2] = idx_k[n];
                    }
                    for(std::size_t w = group_id * in_per_group; w < (group_id + 1) * in_per_group;
                        w++)
                    {
                        idx_in[1]  = w;
                        idx_wei[0] = w;
                        acc += static_cast<double>(input(idx_in.begin(), idx_in.end())) *
                               static_cast<double>(weights(idx_wei.begin(), idx_wei.end()));
                    }
                });
                output[i] = static_cast<type>(acc);
            });
        });
        return result;
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
