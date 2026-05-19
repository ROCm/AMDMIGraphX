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
#ifndef MIGRAPHX_GUARD_OPERATORS_STEP_HPP
#define MIGRAPHX_GUARD_OPERATORS_STEP_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct step
{
    std::vector<int64_t> axes;
    std::vector<int64_t> steps;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.steps, "steps"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "step"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& input = inputs.at(0);

        if(axes.size() != steps.size())
        {
            MIGRAPHX_THROW("STEP: attribute axes {" + to_string_range(axes) +
                           "} has different dimensions from step {" + to_string_range(steps) +
                           "}.");
        }

        if(std::any_of(axes.begin(), axes.end(), [&](auto axis) { return axis >= input.ndim(); }))
        {
            MIGRAPHX_THROW("STEP: axis value is out of range!");
        }

        auto unified  = shape::to_dynamic({input}).front();
        auto dds      = unified.dyn_dims();
        auto dstrides = unified.symbolic() ? unified.dyn_strides() : std::vector<sym::expr>{};
        for(auto i : range(axes.size()))
        {
            auto s       = static_cast<std::size_t>(steps[i]);
            dds[axes[i]] = (dds[axes[i]] + (s - 1)) / s;
            if(unified.symbolic())
                dstrides[axes[i]] = dstrides[axes[i]] * sym::lit(s);
        }
        if(not input.dynamic())
            return shape{input.type(), dds, dstrides}.to_static();
        if(unified.symbolic())
            return shape{input.type(), dds, dstrides};
        return shape{input.type(), dds};
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
