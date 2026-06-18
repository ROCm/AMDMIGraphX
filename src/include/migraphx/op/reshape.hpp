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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/sat_ops.hpp>
#include <migraphx/reshape_dims.hpp>

#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * 1 input version:
 * reshape(input_data)
 * this.dims = output_dims
 * Makes a copy of input_data to the output shape.
 *
 * 2 input version:
 * reshape(input_data, output_buffer)
 * this.dims = unset
 * Copies input_data to output_buffer; output_buffer already has the output shape.
 * This version will not fail gracefully if the input shape and output_buffer shape are
 * incompatible. There's a throw that will catch when the number of elements do not match at
 * runtime. This version should only be used for dynamic reshapes (output dimensions only known at
 * runtime). If output_buffer has a static shape during compile/parse, you can use the 1 input
 * version.
 */
struct reshape
{
    std::vector<dim_like> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    std::string name() const { return "reshape"; }

    // Assumes that the shape from the `dims` attribute will be valid at run-time.
    // Makes no checks for the validity of the `dims` attribute for the given input shape.
    shape dyn_1arg_compute_shape(shape s0) const
    {
        auto input_dyn_dims    = s0.dyn_dims();
        const auto neg_dim_num = std::distance(
            this->dims.begin(), std::find(this->dims.begin(), this->dims.end(), dim_like{-1}));
        const bool has_negative_dim_attr = neg_dim_num < dims.size();
        // construct output dynamic shape from dims attribute
        std::vector<shape::dynamic_dimension> output_dyn_dims(dims.size());
        // NOTE: input_dyn_dims.size() may not equal dims.size()
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
            else
            {
                std::size_t u_dim     = std::get<int64_t>(d);
                output_dyn_dims.at(i) = {u_dim, u_dim};
            }
        }

        if(has_negative_dim_attr)
        {
            // comparing the -1 dimension against the other dimensions

            // accumulate the minimum and maximum elements in the dimensions before the -1 dimension
            std::size_t min_cur_elements = 1;
            std::size_t max_cur_elements = 1;
            for(const auto& dd : output_dyn_dims)
            {
                auto dd_interval = dd.get_interval();
                min_cur_elements = mul_sat(min_cur_elements, dd_interval.min);
                max_cur_elements = mul_sat(max_cur_elements, dd_interval.max);
            }
            // accumulate the elements in the input dimensions
            std::size_t min_input_elements = 1;
            std::size_t max_input_elements = 1;
            for(const auto& dd : input_dyn_dims)
            {
                auto dd_interval   = dd.get_interval();
                min_input_elements = mul_sat(min_input_elements, dd_interval.min);
                max_input_elements = mul_sat(max_input_elements, dd_interval.max);
            }

            // maximum dimensions should never accumulate to zero
            assert(max_cur_elements != 0);

            std::size_t max_int = std::numeric_limits<std::size_t>::max();
            // handle 0 dimension value (keep unknown lower bound)
            std::size_t min_dim =
                (min_cur_elements == 0) ? 0 : min_input_elements / min_cur_elements;
            // handle maximum dimension value (keep unknown upper bound)
            std::size_t max_dim =
                (max_cur_elements == max_int) ? max_int : max_input_elements / max_cur_elements;
            shape::dynamic_dimension x_dd   = {min_dim, max_dim};
            output_dyn_dims.at(neg_dim_num) = x_dd;
        }
        return {s0.type(), output_dyn_dims};
    }

    // Resolves the output dims for static and symbolic input through one path.
    shape symbolic_compute_shape(const shape& s0) const
    {
        // Lift static input to symbolic literals so the same dd arithmetic resolves both.
        auto sym_in          = s0.to_symbolic();
        auto output_dyn_dims = resolve_reshape_dims(sym_in, dims);
        const bool has_inferred_dim =
            std::find(dims.begin(), dims.end(), dim_like{-1}) != dims.end();
        const bool dims_have_symbolic = std::any_of(dims.begin(), dims.end(), is_symbolic);

        // Preserve the input layout when reshape_dims can derive it; else standard.
        std::vector<sym::expr> target(output_dyn_dims.size());
        std::transform(output_dyn_dims.begin(),
                       output_dyn_dims.end(),
                       target.begin(),
                       [](const auto& dd) { return dd.sym_expr; });
        auto result = reshape_dims(sym_in, target, {.lazy = false})
                          .value_or(shape{s0.type(), output_dyn_dims});

        // Validate when the count is fully determined: always for a concrete result;
        // for a symbolic dim/input skip an inferred -1, since N*(6/N) won't fold.
        if(not dims_have_symbolic and (not s0.symbolic() or not has_inferred_dim) and
           result.sym_elements() != s0.sym_elements())
            MIGRAPHX_THROW("Reshape: Wrong number of elements for reshape: reshape has " +
                           to_string(result.sym_elements()) + " elements whereas the input has " +
                           to_string(s0.sym_elements()));

        // Only a static input with integer dims is fully literal; evaluate it back to
        // the concrete layout. Anything symbolic stays symbolic.
        if(not s0.symbolic() and not dims_have_symbolic)
            return result.to_static();
        return result;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);
        if(inputs.size() == 2)
            return inputs.back();

        validate_reshape_dims(name(), dims);

        const auto& s0 = inputs.front();
        if(s0.dynamic() and not s0.symbolic())
        {
            // A symbolic dim has no range interpretation, so it cannot target a
            // range-based input.
            if(std::any_of(dims.begin(), dims.end(), is_symbolic))
                MIGRAPHX_THROW("reshape: range-based input only supports int64 dim entries");
            return dyn_1arg_compute_shape(s0);
        }
        return symbolic_compute_shape(s0);
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        if(args.size() == 1)
        {
            argument result{dyn_out.computed_shape};

            visit_all(result, args[0])([&](auto output, auto input) {
                std::copy(input.begin(), input.end(), output.begin());
            });
            return result;
        }
        else
        {
            // 2 arg
            if(args[0].get_shape().elements() != args[1].get_shape().elements())
            {
                MIGRAPHX_THROW("Reshape: Number of elements must match at runtime. Input: " +
                               std::to_string(args[0].get_shape().elements()) +
                               " Output buffer: " + std::to_string(args[1].get_shape().elements()));
            }
            visit_all(args[1], args[0])([&](auto output, auto input) {
                std::copy(input.begin(), input.end(), output.begin());
            });
            return args[1];
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
