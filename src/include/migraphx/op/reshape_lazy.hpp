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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESHAPE_LAZY_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESHAPE_LAZY_HPP

#include <numeric>
#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/sat_ops.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reshape_lazy
{
    std::vector<dim_like> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    value attributes() const { return {{"require_std_shape", true}}; }

    std::string name() const { return "reshape_lazy"; }

    shape dyn_1arg_compute_shape(shape s0) const
    {
        auto input_dyn_dims    = s0.dyn_dims();
        const auto neg_dim_num = std::distance(
            this->dims.begin(), std::find(this->dims.begin(), this->dims.end(), dim_like{-1}));
        const bool has_negative_dim_attr = neg_dim_num < dims.size();
        std::vector<shape::dynamic_dimension> output_dyn_dims(dims.size());
        for(std::size_t i = 0; i < dims.size(); ++i)
        {
            auto d = dims.at(i);
            if(d == dim_like{0})
            {
                output_dyn_dims.at(i) = input_dyn_dims.at(i);
            }
            else if(d == dim_like{-1})
            {
                output_dyn_dims.at(i) = {1, 1};
            }
            else if(std::holds_alternative<shape::dynamic_dimension>(d))
            {
                output_dyn_dims.at(i) = std::get<shape::dynamic_dimension>(d);
            }
            else
            {
                std::size_t u_dim     = std::get<int64_t>(d);
                output_dyn_dims.at(i) = {u_dim, u_dim};
            }
        }

        if(has_negative_dim_attr)
        {
            std::size_t min_cur_elements = 1;
            std::size_t max_cur_elements = 1;
            for(const auto& dd : output_dyn_dims)
            {
                auto dd_interval = dd.get_interval();
                min_cur_elements = mul_sat(min_cur_elements, dd_interval.min);
                max_cur_elements = mul_sat(max_cur_elements, dd_interval.max);
            }
            std::size_t min_input_elements = 1;
            std::size_t max_input_elements = 1;
            for(const auto& dd : input_dyn_dims)
            {
                auto dd_interval   = dd.get_interval();
                min_input_elements = mul_sat(min_input_elements, dd_interval.min);
                max_input_elements = mul_sat(max_input_elements, dd_interval.max);
            }

            assert(max_cur_elements != 0);

            std::size_t max_int = std::numeric_limits<std::size_t>::max();
            std::size_t min_dim =
                (min_cur_elements == 0) ? 0 : min_input_elements / min_cur_elements;
            std::size_t max_dim =
                (max_cur_elements == max_int) ? max_int : max_input_elements / max_cur_elements;
            shape::dynamic_dimension x_dd   = {min_dim, max_dim};
            output_dyn_dims.at(neg_dim_num) = x_dd;
        }
        return {s0.type(), output_dyn_dims};
    }

    shape dyn_compute_shape(shape s0) const
    {
        const auto& dyn_dims = s0.dyn_dims();
        auto num_not_fixed   = std::count_if(
            dyn_dims.cbegin(), dyn_dims.cend(), [](const auto& dd) { return not dd.is_fixed(); });
        if(num_not_fixed != 1)
        {
            MIGRAPHX_THROW(
                "reshape_lazy: Only supports one non-fixed dynamic_dimension but input {" +
                to_string_range(dyn_dims) + "} has " + to_string(num_not_fixed));
        }
        // track number of fixed elements in input and output
        std::size_t num_dims_ele = 1;
        std::size_t num_dd_ele   = 1;
        for(std::size_t i = 0; i < dyn_dims.size(); ++i)
        {
            if(dyn_dims[i].is_fixed())
            {
                num_dims_ele *= std::get<int64_t>(dims[i]);
                num_dd_ele *= dyn_dims[i].get_interval().min;
            }
            else
            {
                if(dims[i] != dim_like{0} and dims[i] != dim_like{-1})
                {
                    MIGRAPHX_THROW(
                        "reshape_lazy: Non-fixed dynamic_dimension doesn't match with 0 or -1 "
                        "output dimension");
                }
            }
        }
        if(num_dims_ele != num_dd_ele)
        {
            MIGRAPHX_THROW("reshape_lazy: Number of fixed elements must match. Input: " +
                           std::to_string(num_dd_ele) + " Output: " + std::to_string(num_dims_ele));
        }
        // construct output dynamic shape from dims attribute
        std::vector<shape::dynamic_dimension> output_dyn_dims(dims.size());
        std::transform(dims.cbegin(),
                       dims.cend(),
                       dyn_dims.cbegin(),
                       output_dyn_dims.begin(),
                       [](const dim_like& d, auto dyn_dim) {
                           if(not dyn_dim.is_fixed())
                               return dyn_dim;
                           std::size_t dim = std::get<int64_t>(d);
                           return shape::dynamic_dimension{dim, dim};
                       });
        return {s0.type(), output_dyn_dims};
    }

    shape static_compute_shape(std::vector<shape> inputs,
                               const std::vector<dim_like>& rdims_attr,
                               std::size_t n_neg_dims) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(rdims_attr.size());
        std::transform(rdims_attr.begin(), rdims_attr.end(), rdims.begin(), [](const dim_like& d) {
            return std::get<int64_t>(d);
        });

        for(std::size_t i = 0; i < rdims_attr.size(); i++)
        {
            if(rdims_attr[i] == dim_like{0})
                rdims[i] = idims[i];

            if(rdims_attr[i] == dim_like{-1})
                rdims[i] = 1;
        }

        if(n_neg_dims > 0)
        {
            size_t missing_dim =
                inputs.front().elements() /
                std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(rdims_attr[i] == dim_like{-1})
                    rdims[i] = missing_dim;
            }
        }

        auto s = reshape_dims(inputs.front(), rdims, {.lazy = true});
        if(not s.has_value())
            MIGRAPHX_THROW("reshape_lazy on axis that is not packed.");

        if(s->elements() != inputs.front().elements())
            MIGRAPHX_THROW(
                "reshape_lazy: Wrong number of elements for reshape_lazy: reshape_lazy has " +
                std::to_string(s->elements()) + " elements whereas the input has " +
                std::to_string(inputs.front().elements()));

        assert(s->bytes() == inputs.front().bytes());
        return *s;
    }

    shape static_compute_shape(std::vector<shape> inputs, std::size_t n_neg_dims) const
    {
        return static_compute_shape(std::move(inputs), dims, n_neg_dims);
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);

        const bool has_dyn_dim_entries = std::any_of(dims.begin(), dims.end(), [](const auto& d) {
            return std::holds_alternative<shape::dynamic_dimension>(d);
        });

        auto n_neg_dims = std::count(dims.begin(), dims.end(), dim_like{-1});
        if(n_neg_dims > 1)
            MIGRAPHX_THROW("reshape_lazy: Dimensions for reshape_lazy can only have one -1 dim but "
                           "given {" +
                           to_string_range(dims) + "} with " + to_string(n_neg_dims) + " -1 dims");
        const auto& s0 = inputs[0];
        // Static input: resolve fixed dynamic_dimension entries to literals and unfixed to -1.
        if(has_dyn_dim_entries and not s0.dynamic())
        {
            std::vector<dim_like> resolved_dims = dims;
            std::size_t resolved_neg_dims       = n_neg_dims;
            for(auto& d : resolved_dims)
            {
                if(std::holds_alternative<shape::dynamic_dimension>(d))
                {
                    const auto& dd = std::get<shape::dynamic_dimension>(d);
                    if(dd.is_fixed())
                    {
                        d = static_cast<int64_t>(shape::static_dim_value(dd));
                    }
                    else
                    {
                        d = dim_like{-1};
                        ++resolved_neg_dims;
                    }
                }
            }
            if(resolved_neg_dims > 1)
                MIGRAPHX_THROW("reshape_lazy: Dimensions for reshape_lazy can only have one -1 dim "
                               "but given {" +
                               to_string_range(dims) + "} with " + to_string(resolved_neg_dims) +
                               " -1 dims");
            return static_compute_shape(inputs, resolved_dims, resolved_neg_dims);
        }
        if(s0.dynamic())
        {
            if(has_dyn_dim_entries)
                return dyn_1arg_compute_shape(s0);
            return dyn_compute_shape(s0);
        }
        return static_compute_shape(inputs, n_neg_dims);
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        return args[0].reshape(dyn_out.computed_shape);
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
