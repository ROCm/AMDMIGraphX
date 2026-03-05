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

struct dynamic_range : op_name<dynamic_range>
{
    std::size_t max_output = std::numeric_limits<int>::max();

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.max_output, "max_output"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        check_shapes{inputs, *this}.has(3).same_type();
        const auto& type = inputs.at(0).type();
        // The output shape is 1D with unknown size if we don't evaluate.
        return shape{type, {shape::dynamic_dimension{0, max_output}}};
    }
    argument compute(const dyn_output&, std::vector<argument> args) const
    {
        size_t num_elements = 0;
        visit_all(args[0], args[1], args[2])([&](auto start, auto limit, auto delta) {
            auto start_val = start.front();
            auto limit_val = limit.front();
            auto delta_val = delta.front();

            // number_of_elements = max( ceil( (limit - start) / delta ), 0 )
            double num_elements_d = std::ceil(static_cast<double>(limit_val - start_val) /
                                              static_cast<double>(delta_val));
            num_elements          = static_cast<size_t>(std::max(0.0, num_elements_d));
        });

        argument result{shape{args[0].get_shape().type(), {num_elements}}};

        visit_all(args[0], args[2], result)([&](auto start, auto delta, auto output) {
            auto start_val = start.front();
            auto delta_val = delta.front();

            for(size_t i = 0; i < num_elements; ++i)
            {
                output[i] = start_val + (static_cast<decltype(start_val)>(i) * delta_val);
            }
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
