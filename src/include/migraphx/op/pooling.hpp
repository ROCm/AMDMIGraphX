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

    // dimensions of the pooling kernel or window.  Must have size
    // 2 smaller than the input tensor dimensions, since the first
    // two indices are considered
    // non-spatial: batch size and channel
    std::vector<std::size_t> lengths = {1, 1};

    // Size of stride to take from one placement of the pooling kernel to the
    // next.  Usually set the same as lengths so that the kernel tiles over
    // the input with no gaps or overlaps.
    // Must be the same size as lengths.

    // This is distinct from the strides used by the shape class.
    std::vector<std::size_t> stride = {1, 1};

    // The amount each dimension is padded, both before and after.  Can be
    // either the same size as lengths, or twice as long.  When padding
    // is double the length of lengths, the first n values are read as
    // pre-padding for each dimension and the last n values are post-padding.
    // (The latter most commonly used if the total padding desired is an odd number)
    std::vector<std::size_t> padding = {0, 0};

    // Dilations are not supported at this time.

    // ceiling mode flag.  When true, round the size of output up and add
    // padding as necessary to allow placements of the pooling kernel.
    // When false, round down NEW: also clip the pooling window if it extends
    // beyond the input bounds, leaving padding cells out of the calculation.
    // (It's still possible fot the window to start on a padding cell if it
    // extends partially into the input bounds)
    // TODO:  should this be true or false by default?  Should it be set to true whenever any
    // padding is given?
    bool ceil_mode = false;
    int lp_order   = 2;

    // Global pooling with dynamic shape input
    bool dyn_global = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"),
                    f(self.ceil_mode, "ceil_mode"),
                    f(self.lp_order, "lp_order"),
                    f(self.dyn_global, "dyn_global"));
    }

    std::string name() const { return "pooling"; }

    void check_attribute_size() const
    {
        if((padding.size() != stride.size() and (padding.size() / 2) != stride.size()) or
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
            assert(input_lens[i + 2] + padding_factor >= lengths[i]);
            std::size_t dim_size = input_lens[i + 2] + padding_factor - lengths[i];
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
                      Op op) const
    {
        auto in_s    = input.get_shape();
        auto in_lens = in_s.lens();
        // printf("strides ");for(auto aa : stride) std::cout << aa << ", ";    std::cout << "\n";
        // printf("in_lens ");for(auto aa : in_lens) std::cout << aa << ", ";    std::cout << "\n";

        // For each element of output; i.e., for each placement of pooling kernel...
        par_for(output_shape.elements(), [&](auto i) {
            auto idx_o = output_shape.multi(i);
            auto n_dim = idx_o.size();
            // win_start is the starting offset of the pooling window
            std::vector<int> win_start;

            // TODO Thu Jun 8 2023:  if we want to allow reduced-size non-pad windows, then
            //   insert the clipped size into win_size.  if(!ceil_mode)...
            std::vector<std::size_t> win_size;
            // printf("\n");

            // For each spatial dimension, find starting and ending index of pooling kernel
            for(std::size_t dim = 2; dim < n_dim; ++dim)
            {
                auto d_2 = dim - 2;
                int start =
                    static_cast<int>(idx_o[dim] * stride[d_2]) - static_cast<int>(padding[d_2]);

                if(ceil_mode)
                {
                    // In ceiling mode, the pooling kernel always fits, possibly using
                    // padding values
                    win_start.push_back(start);
                    win_size.push_back(kernel_dims[d_2]);
                }
                else
                {
                    // In non-ceiling mode, we clip the pooling kernel at the edges of the input
                    // TODO: what is behavior when ceil_mode is false but padding is given?
                    int end = std::min(start + kernel_dims[d_2], in_lens[dim]);
                    start   = std::max(start, 0);
                    win_start.push_back(start);
                    win_size.push_back(end - start);
                }
            }

            shape win_shape{output_shape.type(), win_size};
            auto pool_size    = win_shape.elements();
            double output_val = op.template init<Type>();

            // for each element in the window...
            shape_for_each(win_shape, [&](auto idx_w) {
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
                    // std::cout <<  " idx  in bounds ";  for(auto aa : idx) std::cout << aa << ",
                    // ";   std::cout << "\n"; std::cout <<  " window locations ";  for(auto aa :
                    // win_start) std::cout << aa << ", ";   std::cout << "\n"; std::cout <<  " in_s
                    // index   " << in_s.index(idx) << " data is " << input[in_s.index(idx)]  <<
                    // "\n";
                    output_val = op(output_val, input[in_s.index(idx)]);
                }
                else
                {
                    // this is a padding element.  Only zero-padding is supported.  Zeroes
                    // don't contribute to average padding total but can play in max or
                    // lpnorm padding.
                    // std::cout <<  " idx out bounds ";  for(auto aa : idx) std::cout << aa << ",
                    // ";   std::cout << "\n";
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
