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
#include <migraphx/sym_argument.hpp>
#include <migraphx/ranges.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

sym_argument::sym_argument(shape s) : m_data(s.element_space()), m_shape(std::move(s)) {}

sym_argument::sym_argument(std::vector<sym::expr> data, shape s)
    : m_data(std::move(data)), m_shape(std::move(s))
{
}

bool sym_argument::empty() const { return m_data.empty(); }

const shape& sym_argument::get_shape() const { return m_shape; }

tensor_view<sym::expr> sym_argument::get()
{
    if(empty())
        return {};
    return make_view(m_shape, m_data.data());
}

tensor_view<const sym::expr> sym_argument::get() const
{
    if(empty())
        return {};
    return make_view(m_shape, m_data.data());
}

bool sym_argument::valid() const
{
    if(empty() or m_shape.type() == shape::tuple_type or not m_shape.computable() or
       m_shape.dynamic() or m_data.size() < m_shape.element_space())
        return false;
    return none_of(get(), [](const auto& expression) { return expression.empty(); });
}

sym_argument sym_argument::reshape(const shape& s) const
{
    if(not valid() or s.type() == shape::tuple_type or not s.computable() or s.dynamic() or
       m_data.size() < s.element_space())
        return {};

    auto result    = *this;
    result.m_shape = s;
    if(not result.valid())
        return {};
    return result;
}

bool operator==(const sym_argument& x, const sym_argument& y)
{
    if(x.get_shape() != y.get_shape() or x.empty() != y.empty())
        return false;
    return x.empty() or x.get() == y.get();
}

bool operator!=(const sym_argument& x, const sym_argument& y) { return not(x == y); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
