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
#ifndef MIGRAPHX_GUARD_OPERATORS_DEREF_HPP
#define MIGRAPHX_GUARD_OPERATORS_DEREF_HPP

#include <migraphx/config.hpp>
#include <migraphx/op/unary.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct deref : unary<deref>
{
    shape::type_t target_type = shape::half_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.target_type, "target_type"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        const auto& input = inputs.at(0);
        return input.with_type(target_type);
    }

    std::string point_op() const
    {
        return "${function:deref}<" + shape::cpp_type(target_type) + ">(${0})";
    }

    struct auto_cast_ptr
    {
        std::size_t x;

        template <class T>
        operator T()
        {
            return reinterpret_cast<T*>(x);
        }
    };

    auto apply() const
    {
        return [](std::size_t x) {
            return auto_cast_ptr{x};
        };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
