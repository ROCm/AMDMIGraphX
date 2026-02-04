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
#ifndef MIGRAPHX_GUARD_OPERATORS_RANGE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RANGE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/name.hpp>
#include <migraphx/dyn_output.hpp>
#include <cmath>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct range : op_name<range>
{
    std::string name() const { return "range"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);

        const auto& type = inputs.at(0).type();
        if(std::any_of(inputs.begin(), inputs.end(), [&](auto s) { return s.type() != type; }))
        {
            MIGRAPHX_THROW("RANGE: valid inputs types are half, float, double, int16, int32, and "
                           "int64, and all inputs must be of the same type");
        }

        // The output shape is 1D with unknown size if we don't evaluate.
        return shape{type, {shape::dynamic_dimension{0, std::numeric_limits<std::size_t>::max()}}};
    }

    argument compute(const dyn_output&, std::vector<argument> args) const
    {
        argument result;
        visit_all(args[0], args[1], args[2])([&](auto start, auto limit, auto delta) {
            auto start_val = start.front();
            auto limit_val = limit.front();
            auto delta_val = delta.front();

            // number_of_elements = max( ceil( (limit - start) / delta ), 0 )
            double num_elements_d = std::ceil(static_cast<double>(limit_val - start_val) /
                                              static_cast<double>(delta_val));
            size_t num_elements   = num_elements_d > 0 ? static_cast<size_t>(num_elements_d) : 0;

            result = argument{shape{args[0].get_shape().type(), {num_elements}}};

            result.visit([&](auto output) {
                for(size_t i = 0; i < num_elements; ++i)
                {
                    output[i] = start_val + (static_cast<decltype(start_val)>(i) * delta_val);
                }
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
