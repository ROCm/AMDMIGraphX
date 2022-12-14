/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONVOLUTION_HPP

#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct convolution
{
    std::vector<std::size_t> padding  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

    int group                   = 1;
    padding_mode_t padding_mode = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.group, "group"),
                    f(self.padding_mode, "padding_mode"));
    }

    std::string name() const { return "convolution"; }

    void check_attribute_size() const
    {
        if((padding.size() != stride.size() and (padding.size() / 2) != stride.size()) or
           stride.size() != dilation.size())
        {
            MIGRAPHX_THROW("CONVOLUTION: inconsistent attribute sizes");
        }
    }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2).same_type().same_ndims().min_ndims(3);
        check_attribute_size();
        // num of dims of input and attribute should match
        const auto input_size   = inputs[0].max_lens().size();
        const auto padding_size = padding.size();

        if(input_size != padding_size / 2 + 2 && input_size != padding_size + 2)
        {
            MIGRAPHX_THROW("CONVOLUTION: input and attribute size mismatch!");
        }

        const shape& x_shape          = inputs.at(0);
        const shape& w_shape          = inputs.at(1);
        const size_t num_spatial_dims = input_size - 2;
        if(num_spatial_dims != this->kdims())
        {
            MIGRAPHX_THROW("CONVOLUTION: input k-dims does not match attribute size");
        }

        if(not x_shape.dynamic() and not w_shape.dynamic() and
           x_shape.lens().at(1) != (w_shape.lens().at(1) * group))
            MIGRAPHX_THROW("CONVOLUTION: mismatched channel numbers");

        if(x_shape.dynamic() or w_shape.dynamic())
        {
            return dynamic_compute_shape(x_shape, w_shape);
        }
        else
        {
            return fixed_compute_shape(x_shape, w_shape);
        }
    }

    std::vector<std::size_t> calc_conv_lens(std::vector<std::size_t> x_lens,
                                            std::vector<std::size_t> w_lens) const
    {
        const size_t num_spatial_dims = x_lens.size() - 2;
        std::vector<size_t> ret       = {};
        // calculate the output shape of the convolution: ((W - K + 2P) / S) + 1
        for(size_t i = 0; i < num_spatial_dims; i++)
        {
            if(x_lens[i] == 0 or w_lens[i] == 0)
            {
                // for handling when a dimension = 0 (opt of dynamic_dimension)
                ret.push_back(0);
            }
            else
            {
                auto padding_factor = 2 * padding[i];
                if(padding.size() == 2 * num_spatial_dims)
                {
                    // when padding is {x0_begin, x1_begin, ... x0_end , x1_end, ...}
                    padding_factor = padding[i] + padding[i + num_spatial_dims];
                }
                ret.push_back(std::size_t(std::max<std::ptrdiff_t>(
                    1,
                    (x_lens[i + 2] - (1 + dilation[i] * (w_lens[i + 2] - 1)) + padding_factor) /
                            stride[i] +
                        1)));
            }
        }
        return ret;
    }

    shape dynamic_compute_shape(shape x_shape, shape w_shape) const
    {
        std::vector<shape::dynamic_dimension> output_dyn_dims = {};

        auto dynamic_shape_push_back = [&](const shape& input_shape) {
            if(input_shape.dynamic())
            {
                output_dyn_dims.push_back(input_shape.dyn_dims().at(0));
            }
            else
            {
                auto l = input_shape.lens().at(0);
                output_dyn_dims.push_back({l, l, 0});
            }
        };

        dynamic_shape_push_back(x_shape);
        dynamic_shape_push_back(w_shape);

        const size_t num_spatial_dims = x_shape.max_lens().size() - 2;
        if(padding_mode != default_)
        {
            for(std::size_t i = 0; i < num_spatial_dims; ++i)
            {
                auto ceil_div = [](std::size_t x, std::size_t y) { return (x + y - 1) / y; };
                auto s        = stride[i];
                if(x_shape.dynamic())
                {
                    auto x = x_shape.dyn_dims()[i + 2];
                    output_dyn_dims.push_back(shape::dynamic_dimension{
                        ceil_div(x.min, s), ceil_div(x.max, s), ceil_div(x.opt, s)});
                }
                else
                {
                    auto od = ceil_div(x_shape.lens()[i + 2], s);
                    output_dyn_dims.push_back(shape::dynamic_dimension{od, od, 0});
                }
            }
        }
        else
        {
            auto min_spatial_dims = calc_conv_lens(x_shape.min_lens(), w_shape.max_lens());
            auto max_spatial_dims = calc_conv_lens(x_shape.max_lens(), w_shape.min_lens());
            auto opt_spatial_dims = calc_conv_lens(x_shape.opt_lens(), w_shape.opt_lens());
            for(size_t i = 0; i < num_spatial_dims; ++i)
            {
                output_dyn_dims.push_back(shape::dynamic_dimension{
                    min_spatial_dims[i], max_spatial_dims[i], opt_spatial_dims[i]});
            }
        }
        return shape{x_shape.type(), output_dyn_dims};
    }

    shape fixed_compute_shape(shape x_shape, shape w_shape) const
    {
        std::vector<size_t> output_lens{x_shape.lens()[0], w_shape.lens()[0]};
        auto spatial_lens = calc_conv_lens(x_shape.lens(), w_shape.lens());
        std::for_each(spatial_lens.begin(), spatial_lens.end(), [&output_lens](auto x) {
            output_lens.push_back(x);
        });
        return x_shape.with_lens(output_lens);
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
