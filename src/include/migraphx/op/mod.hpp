/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_MUL_HPP
#define MIGRAPHX_GUARD_OPERATORS_MUL_HPP

#include <array>
#include <migraphx/op/binary.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <typename T>
T mod_op(T x, T y)
{
    return (x % y);
}

template <>
float mod_op<float>(float x, float y)
{
    return std::fmod(x, y);
}

template <>
double mod_op<double>(double x, double y)
{
    return std::fmod(x, y);
}

template <>
half_float::half mod_op<half_float::half>(half_float::half x, half_float::half y)
{
    return half_float::fmod(x, y);
}

struct mod : binary<mod>
{
    bool fmod_flag;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.fmod_flag, "fmod_flag"));
    }

    value attributes() const
    {
        auto a         = base_attributes();
        a["fmod_flag"] = fmod_flag;
        return a;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, (*this)}.has(2).same_type().same_dims();
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);

        if((s0.type() == shape::float_type || s0.type() == shape::double_type ||
            s0.type() == shape::half_type) &&
           (fmod_flag == false))
        {
            MIGRAPHX_THROW("fmod must be true for floating data types");
        }

        if(s0 == s1 and s0.packed())
        {
            return s0;
        }
        else if(s0.packed() != s1.packed())
        {
            return s0.packed() ? s0 : s1;
        }
        else if(s0.broadcasted() != s1.broadcasted())
        {
            return s0.broadcasted() ? s1.with_lens(s0.lens()) : s0.with_lens(s0.lens());
        }
        else
        {
            return {s0.type(), s0.lens()};
        }
    }

    std::string point_function() const { return "mod"; }
    auto apply() const
    {
        return [&](auto x, auto y) { return mod_op<decltype(x)>(x, y); };
    }

    mod(bool fmod = false) : fmod_flag{fmod} {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
