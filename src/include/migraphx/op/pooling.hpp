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
#include <migraphx/pad_calc.hpp>
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
    /**
     * The Pooling operator mostly follows the specifications for the Onnx pooling op.
     * Inputs are assumed to have 3 or more dimensions, of which the first 2 are
     * considered "non-spatial," and pooling only takes place over the other dimensions
     * Dimension 0 is assumed to be batch size and dimension 1 is channel count.  Thus
     * for a set of standard RGB images the input would have 4 dimensions:
     *
     * dimension 0 - image # in batch
     * dimension 1 - color channel r, g, or b
     * dimension 2 - row
     * dimension 3 - column
     *
     * Pooling attributes padding, stride, lengths are vectors matching the spatial dimensions,
     * i.e. 2 dimensions smaller than the input.
     *
     * TODO:  dilation is not implemented at the time of writing.
     */
    pooling_mode mode = {pooling_mode::average};

    // Amount of zero-padding to add before and after each (spatial) dimension to
    // make pooling window fit in the data area.  padding can also have size
    // double the number of dimensions, in which case padding[i] and padding[i+lengths.size()]
    // are the padding before and after the i'th dimension.
    std::vector<std::size_t> padding = {0, 0};
    // Number of elements to step, in each dimension, from one pooling window to the next.
    // If stride is smaller than lengths, pooling windows will overlap.
    std::vector<std::size_t> stride = {1, 1};
    // dimensions of the pooling window, or kernel_shape.
    std::vector<std::size_t> lengths = {1, 1};
    bool ceil_mode                   = false;
    int lp_order                     = 2;

    // Mode for auto padding.  default_ indicates no auto padding.
    padding_mode_t padding_mode = padding_mode_t::default_;

    // Global pooling with dynamic shape input
    bool dyn_global = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"),
                    f(self.ceil_mode, "ceil_mode"),
                    f(self.lp_order, "lp_order"),
                    f(self.dyn_global, "dyn_global"));
    }

    std::string name() const { return "pooling"; }

    void check_attribute_size() const
    {
        if((padding_mode == default_ and padding.size() != stride.size() and
            (padding.size()) != stride.size() * 2) or
           (not dyn_global and stride.size() != lengths.size()))
        {
            MIGRAPHX_THROW("POOLING: inconsistent attribute sizes");
        }
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    std::vector<std::size_t> calc_spatial_dim_out(const std::vector<std::size_t>& input_lens,
                                                  std::size_t kdims) const
    {
        std::vector<std::size_t> output_lens{};
        for(size_t i = 0; i < kdims; ++i)
        {
            std::size_t padding_factor = 2 * padding[i];
            if(padding.size() == 2 * kdims)
                padding_factor = padding[i] + padding[i + kdims];
            std::size_t dim_size;
            if(input_lens[i + 2] + padding_factor < lengths[i])
            {
                if(padding_mode == default_)
                    MIGRAPHX_THROW(
                        "POOLING: given padding too little for these padding window lengths");
                // lengths can be legitimately larger only if we're doing auto padding
                // with a dynamic shape, in which case given padding is ignored.  Set a dummy value;
                dim_size = 2;
            }
            else
            {
                dim_size = input_lens[i + 2] + padding_factor - lengths[i];
            }
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
        if(input.ndim() != padding_size / 2 + 2 and input.ndim() != padding_size + 2)
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
            else if(padding_mode != default_)
            {
                const size_t num_spatial_dims = inputs[0].ndim() - 2;
                const shape& x_shape          = inputs[0];
                // this is convolution::dynamic_compute_shape()

                for(std::size_t i = 0; i < num_spatial_dims; ++i)
                {
                    auto ceil_div = [](std::size_t x, std::size_t y) { return (x + y - 1) / y; };
                    auto s        = stride[i];
                    if(x_shape.dynamic()) // redundant check here
                    {
                        auto x = x_shape.dyn_dims()[i + 2];
                        std::set<std::size_t> optimals{};
                        std::transform(x.optimals.begin(),
                                       x.optimals.end(),
                                       std::inserter(optimals, optimals.begin()),
                                       [&](auto o) { return ceil_div(o, s); });
                        output_dyn_dims.push_back(shape::dynamic_dimension{
                            ceil_div(x.min, s), ceil_div(x.max, s), optimals});
                    }
                    else
                    {
                        MIGRAPHX_THROW("debug: how did this happen?");
                        auto od = ceil_div(x_shape.lens()[i + 2], s);
                        output_dyn_dims.push_back(shape::dynamic_dimension{od, od});
                    }
                }
                return {input.type(), output_dyn_dims};
            }
            else
            {
                // does not compute for optimals
                auto min_spatial_dims = calc_spatial_dim_out(input.min_lens(), kdims);
                auto max_spatial_dims = calc_spatial_dim_out(input.max_lens(), kdims);
                for(size_t i = 0; i < kdims; ++i)
                {
                    output_dyn_dims.push_back(
                        shape::dynamic_dimension{min_spatial_dims[i], max_spatial_dims[i], {}});
                }
                shape out_shape(input.type(), output_dyn_dims);
                // out_shape.debug_print();
                return {input.type(), output_dyn_dims};
            }
        }
        else
        {
            auto input_lens = input.lens();

            std::vector<std::size_t> output_lens(input_lens.begin(), input_lens.begin() + 2);
            // Used for when normalize_compute_shape() is called again at model eval time
            // for an originally dynamic shape. Since kernel shape is not used with dyn_global.
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

        double final(double x, std::size_t) const { return std::pow(x, 1. / p); }
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
                      const std::vector<std::size_t>& padding_vals,
                      Op op) const
    {
        auto in_s    = input.get_shape();
        auto in_lens = in_s.lens();
        par_for(output_shape.elements(), [&](auto i) {
            auto idx_o = output_shape.multi(i);
            auto n_dim = idx_o.size();
            std::vector<std::size_t> win_start;
            std::vector<std::size_t> win_size;
            for(std::size_t dim = 2; dim < n_dim; ++dim)
            {
                auto d_2 = dim - 2;

                // TODO:  This lambda pads the pooling window with zeroes, but by dynamically sizing
                // win_shape it ignores them when doing the calculation.  It avoids giving index
                // out-of-bounds errors but gives incorrect results for edge windows that include
                // padding.  For example, a 2x2 window at a corner value that contains a 1 and three
                // padded 0's should give a value of 0.25 for average pooling, but instead
                // returns 1.

                int start = static_cast<int>(idx_o[dim] * stride[d_2]) -
                            static_cast<int>(padding_vals[d_2]);
                int end = std::min(start + kernel_dims[d_2], in_lens[dim]);
                start   = std::max(start, 0);
                win_start.push_back(start);
                win_size.push_back(end - start);
            }

            shape win_shape{output_shape.type(), win_size};

            auto pool_size    = win_shape.elements();
            double output_val = op.template init<Type>();
            shape_for_each(win_shape, [&](auto idx_w) {
                auto idx = idx_o;
                std::transform(idx_w.begin(),
                               idx_w.end(),
                               win_start.begin(),
                               idx.begin() + 2,
                               [](auto ii, auto jj) { return ii + jj; });
                if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                   idx < in_lens)
                {
                    output_val = op(output_val, input[in_s.index(idx)]);
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
        shape output_shape;
        // If we have to auto-calculate padding, it will be passed to calc_pooling() as an argument
        // instead of the member variable padding.
        std::vector<std::size_t> temp_padding(padding);
        if(dyn_global)
        {
            // for dynamic GlobalPooling, there's no padding
            kernel_dims.insert(kernel_dims.end(), input_lens.begin() + 2, input_lens.end());
            output_shape = dyn_out.computed_shape;
        }
        else if((padding_mode != op::padding_mode_t::default_))
        {
            // if padding_mode is set, input was a dynamic size.  Calculate padded size now.

            // wei_lens is the same as kernel_dims, but prepended with the 2 non-
            // spatial dimensions.  For size computations, it's used like the weights
            // tensor for convolutions.
            std::vector<std::size_t> wei_lens;
            wei_lens.insert(wei_lens.end(), input_lens.begin(), input_lens.begin() + 2);
            wei_lens.insert(wei_lens.end(), lengths.begin(), lengths.end());
            kernel_dims = this->lengths;

            auto type = args[0].get_shape().type();
            // dilation not currently supported for pooling, so default to all 1's
            temp_padding = calc_dyn_auto_pad(
                input_lens, wei_lens, stride, {1, 1}, bool(padding_mode == op::same_upper));

            output_shape = compute_padded_pool_shape(
                args[0].get_shape(), shape(type, kernel_dims), temp_padding, stride, {1, 1});

            result = argument(output_shape);
        }
        else // fixed/static input
        {
            kernel_dims  = this->lengths;
            output_shape = dyn_out.computed_shape;
        }

        // Perform the computation and populate result
        visit_all(result, args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            switch(mode)
            {
            case migraphx::op::pooling_mode::average:
                calc_pooling<type>(
                    output_shape, output, input, kernel_dims, temp_padding, avg_pool{});
                break;
            case migraphx::op::pooling_mode::max:
                calc_pooling<type>(
                    output_shape, output, input, kernel_dims, temp_padding, max_pool{});
                break;
            case migraphx::op::pooling_mode::lpnorm:
                calc_pooling<type>(
                    output_shape, output, input, kernel_dims, temp_padding, lpnorm_pool{lp_order});
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
