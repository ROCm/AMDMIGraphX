/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_BROADCAST_WITH_DIMS_HPP
#define MIGRAPHX_GUARD_OPERATORS_BROADCAST_WITH_DIMS_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Broadcast the input tensor to the shape defined by the values of the second input.
 * Used as `broadcast_with_dims(input_tensor, dims)`, where dims is a vector of integer dimensions.
 * `input_tensor` must be broadcastable with `dims`, otherwise this operator with throw at compute.
 * This operator can be replaced with `multibroadcast(input_tensor)` if the `dims` vector is
 * constant.
 */
struct broadcast_with_dims
{
    std::string name() const { return "broadcast_with_dims"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this, true}.has(2);
        // check that second input has a static shape
        migraphx::check_shapes{inputs.begin() + 1, inputs.end(), *this, false};
        // output tensor rank greater of input_tensor rank or length of dims vector
        auto input_tensor_shape = inputs.at(0);
        auto dims_shape         = inputs.at(1);
    std:
        size_t out_ndim     = std::max(input_tensor_shape.ndim(), dims_shape.lens().at(0));
        std::size_t max_int = std::numeric_limits<std::size_t>::max();
        std::vector<shape::dynamic_dimension> dyn_dims(out_ndim,
                                                       shape::dynamic_dimension{0, max_int});
        return {input_tensor_shape.type(), dyn_dims};
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        auto in_lens = args.at(0).get_shape().lens();
        std::vector<std::size_t> dims_input(output_shape.ndim());
        args.at(1).visit([&](auto a) { dims_input.assign(a.begin(), a.end()); });
        auto out_lens = compute_broadcasted_lens(in_lens, dims_input);

        // same code as in multibroadcast
        if(s0.ndim() > output_lens.size())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should <= output size");
        }

        auto offset = output_lens.size() - s0.ndim();
        for(std::ptrdiff_t i = s0.ndim() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] != s0.lens()[i] and s0.lens()[i] != 1)
            {
                MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(s0.lens()) +
                               "} cannot be broadcasted to {" + to_string_range(output_lens) +
                               "}!");
            }
        }

        return make_bcast_shape(s0, output_lens, offset);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
