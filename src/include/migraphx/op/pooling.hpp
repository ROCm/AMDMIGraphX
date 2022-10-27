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
#ifndef MIGRAPHX_GUARD_OPERATORS_POOLING_HPP
#define MIGRAPHX_GUARD_OPERATORS_POOLING_HPP

#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/int_divide.hpp>
#include <cmath>
#include <utility>

namespace migraphx {

inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pooling
{
    pooling_mode mode                = {pooling_mode::average};
    std::vector<std::size_t> padding = {0, 0};
    std::vector<std::size_t> stride  = {1, 1};
    std::vector<std::size_t> lengths = {1, 1};
    bool ceil_mode                   = false;
    int lp_order                     = 2;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"),
                    f(self.ceil_mode, "ceil_mode"),
                    f(self.lp_order, "lp_order"));
    }

    std::string name() const { return "pooling"; }

    void check_attribute_size() const
    {
        if((padding.size() != stride.size() and (padding.size() / 2) != stride.size()) or
           stride.size() != lengths.size())
        {
            MIGRAPHX_THROW("POOLING: inconsistent attribute sizes");
        }
    }

    value attributes() const { return {{"normalize_padding", "padding"}}; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);

        const shape& input = inputs.at(0);

        auto input_lens   = input.lens();
        size_t kdims      = input_lens.size() - 2;
        auto input_size   = inputs[0].lens().size();
        auto padding_size = padding.size();
        if(input_size != padding_size / 2 + 2 and input_size != padding_size + 2)
        {
            MIGRAPHX_THROW("POOLING: input and attribute size mismatch!");
        }

        std::vector<std::size_t> output_lens(input_lens.begin(), input_lens.begin() + 2);

        for(size_t i = 0; i < kdims; i++)
        {
            std::ptrdiff_t dim_size;
            auto padding_factor = 2 * padding[i];
            if(padding_size == 2 * kdims)
                padding_factor = padding[i] + padding[i + kdims];
            dim_size = input_lens[i + 2] + padding_factor - lengths[i];
            assert(dim_size >= 0);
            std::size_t len = (ceil_mode) ? ceil_divide<std::ptrdiff_t>(dim_size, stride[i])
                                          : floor_divide<std::ptrdiff_t>(dim_size, stride[i]);

            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(1, len + 1)));
        }
        return inputs[0].with_lens(output_lens);
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
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
    void calc_pooling(const shape& output_shape, Out& output, const In& input, Op op) const
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
                int start =
                    static_cast<int>(idx_o[dim] * stride[d_2]) - static_cast<int>(padding[d_2]);
                int end = std::min(start + lengths[d_2], in_lens[dim]);
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

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            switch(mode)
            {
            case migraphx::op::pooling_mode::average:
                calc_pooling<type>(output_shape, output, input, avg_pool{});
                break;
            case migraphx::op::pooling_mode::max:
                calc_pooling<type>(output_shape, output, input, max_pool{});
                break;
            case migraphx::op::pooling_mode::lpnorm:
                calc_pooling<type>(output_shape, output, input, lpnorm_pool{lp_order});
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
