/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <numeric>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/sat_ops.hpp>

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
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    std::string name() const { return "reshape"; }

    shape dyn_1arg_compute_shape(shape s0) const
    {
        const auto& input_dyn_dims = s0.dyn_dims();
        std::vector<shape::dynamic_dimension> rdims(dims.size());

        for(std::size_t i = 0; i < dims.size(); ++i)
        {
            if(dims[i] == 0)
                rdims[i] = input_dyn_dims.at(i);
            else if(dims[i] == -1)
                rdims[i] = shape::dynamic_dimension{1, 1};
            else
            {
                auto v = static_cast<std::size_t>(dims[i]);
                rdims[i] = shape::dynamic_dimension{v, v};
            }
        }

        auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
        if(n_neg_dims > 0)
        {
            auto total_elements = std::accumulate(
                input_dyn_dims.begin(), input_dyn_dims.end(), shape::dynamic_dimension{1, 1}, std::multiplies<>{});
            auto known_elements = std::accumulate(
                rdims.begin(), rdims.end(), shape::dynamic_dimension{1, 1}, std::multiplies<>{});
            auto missing_dim = total_elements / known_elements;
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(dims[i] == -1)
                    rdims[i] = missing_dim;
            }
        }

        return {s0.type(), rdims};
    }

    shape static_compute_shape(std::vector<shape> inputs, std::size_t n_neg_dims) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(dims.begin(), dims.end());

        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];

            // convert -1 to 1 for rdims since rdims uses size_t (-1 is max_int for size_t)
            if(dims[i] == -1)
                rdims[i] = 1;
        }

        if(n_neg_dims > 0)
        {
            size_t missing_dim =
                inputs.front().elements() /
                std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(dims[i] == -1)
                    rdims[i] = missing_dim;
            }
        }

        auto s = shape{inputs.front().type(), rdims};

        if(s.elements() != inputs.front().elements())
            MIGRAPHX_THROW("Reshape: Wrong number of elements for reshape: reshape has " +
                           std::to_string(s.elements()) + " elements whereas the input has " +
                           std::to_string(inputs.front().elements()));

        return s;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);

        auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
        if(n_neg_dims > 1)
            MIGRAPHX_THROW("Reshape: Dimensions for reshape can only have one -1 dim");

        const auto& s0 = inputs.front();
        if(inputs.size() == 1)
        {
            if(s0.dynamic())
            {
                return dyn_1arg_compute_shape(s0);
            }
            else
            {
                return static_compute_shape(inputs, n_neg_dims);
            }
        }
        else
        {
            return inputs.back();
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        assert(dyn_out.computed_shape.standard());
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
