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
#ifndef MIGRAPHX_GUARD_OPERATORS_WHERE_HPP
#define MIGRAPHX_GUARD_OPERATORS_WHERE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/sym_argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct where
{
    std::string name() const { return "where"; }

    value attributes() const { return {{"pointwise", true}, {"point_op", "${0} ? ${1} : ${2}"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes shape_checker{inputs, *this, true};
        shape_checker.has(3);
        if(auto s = inputs[0]; not s.dynamic() and s.elements() == 1)
            check_shapes{std::next(inputs.begin()), inputs.end(), *this, true}.same_dims();
        else
            shape_checker.same_dims();

        auto s1 = inputs.at(1);
        auto s2 = inputs.at(2);
        // Range-based dynamic (or mixed dynamic/static) inputs only support strict equality.
        if((s1.dynamic() or s2.dynamic()) and not(s1.symbolic() and s2.symbolic()))
        {
            if(s1 == s2)
                return s1;
            MIGRAPHX_THROW("WHERE: dynamic input shapes must be the same but given " +
                           to_string(s1) + " and " + to_string(s2));
        }

        // Compare two static (or two symbolic) shapes, returning a standard shape
        if(s1 == s2 and s1.packed())
        {
            return s1;
        }
        else if(s1.packed() != s2.packed())
        {
            return s1.packed() ? s1 : s2;
        }
        else if(s1.broadcasted() != s2.broadcasted())
        {
            if(s1.symbolic())
                return s1.broadcasted() ? s2.with_lens(s1.dyn_dims()) : s1.with_lens(s1.dyn_dims());
            return s1.broadcasted() ? s2.with_lens(s1.lens()) : s1.with_lens(s1.lens());
        }
        else
        {
            if(s1.symbolic())
                return {s1.type(), s1.dyn_dims()};
            return {s1.type(), s1.lens()};
        }
    }

    sym_argument symbolic_compute(const shape& output_shape,
                                  const std::vector<sym_argument>& args) const
    {
        if(args.size() != 3 or any_of(args, [](const auto& arg) { return arg.empty(); }) or
           args[1].get_shape().lens() != output_shape.lens() or
           args[2].get_shape().lens() != output_shape.lens())
            return {};
        const bool scalar_condition = args[0].get_shape().elements() == 1;
        if(not scalar_condition and args[0].get_shape().lens() != output_shape.lens())
            return {};

        sym_argument result{output_shape};
        const auto condition = args[0].get();
        const auto x         = args[1].get();
        const auto y         = args[2].get();
        auto output          = result.get();
        for(auto i : range(output_shape.elements()))
        {
            const auto condition_value =
                sym::fixed_value(condition[scalar_condition ? 0 : i]);
            if(not condition_value.has_value())
                return {};
            output[i] = sym::to<int64_t>(*condition_value) != 0 ? x[i] : y[i];
        }
        return result;
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        if(auto s = args[0].get_shape(); not s.dynamic() and s.elements() == 1)
            return args[args[0].at<bool>() ? 1 : 2].copy();

        if(output_shape.dynamic())
            output_shape = compute_shape(to_shapes(args));
        argument result{output_shape};

        visit_all(result, args[1], args[2])([&](auto output, const auto x, const auto y) {
            args[0].visit([&](const auto condition) {
                par_for(output_shape.elements(),
                        [&](auto i) { output[i] = condition[i] ? x[i] : y[i]; });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
