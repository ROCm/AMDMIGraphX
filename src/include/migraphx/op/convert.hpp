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
#include <migraphx/sym_argument.hpp>
#include <migraphx/functional.hpp>
#include <cmath>
#include <unordered_map>
#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct convert : unary<convert>
{
    static constexpr bool enable_symbolic_compute = true;

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

    bool supports_symbolic_compute(const shape& output_shape,
                                   const std::vector<sym_argument>& args) const
    {
        auto is_supported = [](shape::type_t t) {
            return t == shape::int64_type or t == shape::int32_type or t == shape::bool_type or
                   t == shape::float_type or t == shape::double_type;
        };
        return args.size() == 1 and is_supported(args[0].get_shape().type()) and
               output_shape.type() == target_type and is_supported(target_type);
    }

    // A floating-point value converted to an integer keeps its expression only when it is a
    // whole number: its literals, the only part that can hold a fraction, have to be whole
    static sym::expr to_integer(const sym::expr& x)
    {
        std::unordered_map<sym::expr, sym::expr> whole;
        bool fraction = false;
        fix([&](auto self, const sym::expr& e) -> void {
            if(e.name() == "literal")
            {
                const auto value = sym::fixed_value(e);
                if(not value.has_value() or std::holds_alternative<int64_t>(*value))
                    return;
                const auto d = sym::to<double>(*value);
                if(std::trunc(d) != d)
                    fraction = true;
                else
                    whole.emplace(e, sym::lit(static_cast<int64_t>(d)));
                return;
            }
            for(const auto& child : e.children())
                self(child);
        })(x);
        if(fraction)
            return {};
        return whole.empty() ? x : x.subs(whole);
    }

    auto apply() const
    {
        return [target = target_type](const sym::expr& x) {
            // Integers up to 2^24 (2^53 for double) are exact as floating point
            if(target == shape::float_type)
                return sym::fits_range(x, -(int64_t{1} << 24), int64_t{1} << 24) ? x : sym::expr{};
            if(target == shape::double_type)
                return sym::fits_range(x, -(int64_t{1} << 53), int64_t{1} << 53) ? x : sym::expr{};
            const auto whole = to_integer(x);
            if(whole.empty())
                return whole;
            if(target == shape::int64_type)
                return whole;
            if(target == shape::int32_type)
                return sym::fits<int32_t>(whole) ? whole : sym::expr{};
            const auto value = sym::fixed_value(whole);
            if(not value.has_value())
                return sym::expr{};
            const auto integer = sym::to<int64_t>(*value);
            if(integer != 0 and integer != 1)
                return sym::expr{};
            return whole;
        };
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
