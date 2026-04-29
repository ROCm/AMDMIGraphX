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

#include <migraphx/dim_like.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>

#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::ostream& operator<<(std::ostream& os, const dim_like& d)
{
    std::visit([&](const auto& x) { os << x; }, d.value);
    return os;
}

migraphx::value dim_like::to_value() const
{
    return std::visit([](const auto& x) { return migraphx::to_value(x); }, this->value);
}

void dim_like::from_value(const migraphx::value& v)
{
    // Backward-compatible path: integer-valued entries (signed or unsigned)
    // route through the int alternative so old .mxr files and call sites that
    // pass plain integer arrays both decode without going through the
    // dynamic_dimension reflect path.
    if(v.is_int64() or v.is_uint64())
    {
        this->value = v.to<int64_t>();
        return;
    }
    shape::dynamic_dimension d;
    migraphx::from_value(v, d);
    this->value = std::move(d);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
