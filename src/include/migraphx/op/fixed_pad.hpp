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
#ifndef MIGRAPHX_GUARD_OPERATORS_FIXED_PAD_HPP
#define MIGRAPHX_GUARD_OPERATORS_FIXED_PAD_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/clamp.hpp>
#include <migraphx/config.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Pads a dynamic input up to a target size, filling the pad with `value`.
 * With no `dims`: target is the input's max dims (no-op on a static input).
 * With `dims`: target is those (possibly symbolic) per-axis dims.
 */
struct fixed_pad
{
    std::vector<sym::expr> dims = {};
    float value                 = 0.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"), f(self.value, "value"));
    }

    std::string name() const { return "fixed_pad"; }

    shape target_shape(const shape& s) const
    {
        assert(dims.size() == s.ndim());
        std::vector<shape::dynamic_dimension> dds(dims.size());
        std::transform(dims.begin(), dims.end(), dds.begin(), [](const auto& e) {
            return shape::dynamic_dimension{e};
        });
        shape result{s.type(), std::move(dds)};
        return result.is_fixed() ? result.to_static() : result;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& s0 = inputs.front();
        if(not dims.empty())
            return target_shape(s0);
        if(s0.dynamic())
        {
            return {s0.type(), s0.max_lens()};
        }
        return s0;
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto& input_arg = args.front();
        auto input_shape      = input_arg.get_shape();
        if(input_shape == output_shape)
            return input_arg;

        argument out{output_shape};
        visit_all(out, input_arg)([&](auto output, auto input) {
            using type = typename decltype(output)::value_type;
            std::fill(output.begin(), output.end(), pad_clamp<type>(value));
            par_for(input_shape.elements(), [&](auto i) {
                auto idx    = input_shape.multi(i);
                output[idx] = input[idx];
            });
        });

        return out;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
