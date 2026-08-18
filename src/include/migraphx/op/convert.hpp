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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONVERT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONVERT_HPP

#include <migraphx/config.hpp>
#include <migraphx/op/unary.hpp>
#include <migraphx/symbolic_tensor_value.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct convert : unary<convert>
{
    shape::type_t target_type = shape::half_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.target_type, "target_type"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& input = inputs.at(0);
        if(input.symbolic())
        {
            return {target_type, input.dyn_dims(), input.dyn_strides()};
        }
        else if(input.dynamic())
        {
            return {target_type, input.dyn_dims()};
        }
        else
        {
            return {target_type, input.lens(), input.strides()};
        }
    }

    std::optional<symbolic_tensor_value>
    symbolic_compute(const shape& output_shape,
                     const std::vector<shape>& input_shapes,
                     const std::vector<std::optional<symbolic_tensor_value>>& input_values) const
    {
        if(input_shapes.size() != 1 or input_shapes.front().type() != shape::int64_type)
            return std::nullopt;
        if(target_type == shape::int64_type)
            return pass_through_symbolic_value(output_shape, input_values);
        if(target_type != shape::bool_type or input_values.size() != 1 or
           not input_values.front().has_value())
            return std::nullopt;
        const auto fixed_values = fixed_integers(*input_values.front());
        if(not fixed_values.has_value() or
           any_of(*fixed_values, [](auto value) { return value != 0 and value != 1; }))
            return std::nullopt;
        return pass_through_symbolic_value(output_shape, input_values);
    }

    std::string point_op() const
    {
        return "${function:convert}<" + shape::cpp_type(target_type) + ">(${0})";
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                using output_type = typename decltype(output)::value_type;
                par_transform(
                    input.begin(), input.end(), output.begin(), [](auto x) -> output_type {
                        double dx = x;
                        if(std::isnan(dx))
                            return std::numeric_limits<output_type>::quiet_NaN();
                        if(not std::is_integral<output_type>{} and std::isinf(dx))
                            return output_type(x);
                        if(dx >= std::numeric_limits<output_type>::max())
                            return std::numeric_limits<output_type>::max();
                        if(dx <= std::numeric_limits<output_type>::lowest())
                            return std::numeric_limits<output_type>::lowest();
                        return output_type(dx);
                    });
            });
        });
        return result;
    }

    convert(shape::type_t t) : target_type{t} {}
    convert() {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
