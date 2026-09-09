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
#ifndef MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/sym_argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct squeeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "squeeze"; }

    // Drops the size-1 axes, preserving the kept axes' strides, for static and
    // symbolic input through one path.
    shape symbolic_compute_shape(const shape& s) const
    {
        auto sym_in       = s.to_symbolic();
        const auto& dds   = sym_in.dyn_dims();
        const auto& strds = sym_in.dyn_strides();
        auto one          = sym::lit(1);
        // A dropped axis must be provably 1.
        if(std::any_of(axes.begin(), axes.end(), [&](auto axis) {
               return not(dds.at(axis).sym_expr == one);
           }))
            MIGRAPHX_THROW("SQUEEZE: axis dimension should be equal to 1; axes {" +
                           to_string_range(axes) + "} of input " + to_string(s));
        std::vector<shape::dynamic_dimension> new_dds;
        std::vector<sym::expr> new_strides;
        for(auto i : range(dds.size()))
        {
            const bool drop = axes.empty() ? (dds[i].sym_expr == one)
                                           : (std::find(axes.begin(), axes.end(), i) != axes.end());
            if(not drop)
            {
                new_dds.push_back(dds[i]);
                new_strides.push_back(strds[i]);
            }
        }
        shape result = new_dds.empty() ? shape{s.type()} : shape{s.type(), new_dds, new_strides};
        if(not s.symbolic())
            return result.to_static();
        return result;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto input_shape = inputs[0];
        if(input_shape.dynamic() and not input_shape.symbolic())
        {
            // Allow for any dynamic_dimension that intersects with {1, 1}.
            // Assuming that the shape at run-time will be compatible.
            if(std::any_of(axes.begin(), axes.end(), [&](auto axis) {
                   return not input_shape.dyn_dims()
                                  .at(axis)
                                  .intersection(shape::dynamic_dimension{1, 1})
                                  .has_value();
                   ;
               }))
            {
                MIGRAPHX_THROW("SQUEEZE: dynamic axis dimension should have an intersection with "
                               "{1, 1}; axes {" +
                               to_string_range(axes) + "} of input " + to_string(input_shape));
            }
            std::vector<shape::dynamic_dimension> dyn_dims = {};
            if(axes.empty())
            {
                std::copy_if(input_shape.dyn_dims().cbegin(),
                             input_shape.dyn_dims().cend(),
                             std::back_inserter(dyn_dims),
                             [&](const auto& dd) { return dd != 1; });
            }
            else
            {
                for(auto i : range(input_shape.ndim()))
                {
                    if(std::find(axes.begin(), axes.end(), i) == axes.end())
                    {
                        dyn_dims.push_back(input_shape.dyn_dims()[i]);
                    }
                }
            }
            return {input_shape.type(), dyn_dims};
        }
        return symbolic_compute_shape(input_shape);
    }

    sym_argument symbolic_compute(const shape& output_shape,
                                  const std::vector<sym_argument>& args) const
    {
        if(args.size() != 1)
            return {};
        return args[0].reshape(output_shape);
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
