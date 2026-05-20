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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_DIM_LIKE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_DIM_LIKE_HPP

#include <cstdint>
#include <ostream>
#include <type_traits>

#include <migraphx/config.hpp>
#include <migraphx/picked_variant.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

// Routes any integral type through int64_t so call sites don't need casts.
struct dim_like_picker
{
    template <class T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
    static int64_t apply(T v)
    {
        return static_cast<int64_t>(v);
    }

    static shape::dynamic_dimension apply(shape::dynamic_dimension d) { return d; }
};

// A dim attribute entry that may be either a plain int64_t or a dynamic_dimension.
using dim_like = picked_variant<dim_like_picker, int64_t, shape::dynamic_dimension>;

// Templated to hide from ADL on unrelated types: a non-template overload would
// be probed during overload resolution for things like vector<dim_like>, which
// would instantiate Picker::apply(vector<...>) and hard-fail.
template <class T, MIGRAPHX_REQUIRES(std::is_same<T, dim_like>{})>
inline std::ostream& operator<<(std::ostream & os, const T & d)
{
    visit([&](const auto& x) { os << x; }, d);
    return os;
}

MIGRAPHX_EXPORT void migraphx_to_value(value& v, const dim_like& d);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, dim_like& d);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
