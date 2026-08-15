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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_SYMBOLIC_TENSOR_VALUE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_SYMBOLIC_TENSOR_VALUE_HPP

#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Exact flattened values of an integral ONNX tensor.
using symbolic_tensor_value = std::vector<sym::expr>;

template <class T>
inline bool can_broadcast_value(const std::vector<T>& value, std::size_t output_size)
{
    return value.size() == output_size or value.size() == 1;
}

template <class T>
inline const T& broadcast_value_at(const std::vector<T>& value, std::size_t index)
{
    return value.at(value.size() == 1 ? 0 : index);
}

template <class F>
std::optional<symbolic_tensor_value>
broadcast_symbolic_values(const symbolic_tensor_value& x, const symbolic_tensor_value& y, F f)
{
    const auto size = std::max(x.size(), y.size());
    if(not can_broadcast_value(x, size) or not can_broadcast_value(y, size))
        return std::nullopt;

    symbolic_tensor_value result;
    result.reserve(size);
    transform(range(size), std::back_inserter(result), [&](auto i) {
        return f(broadcast_value_at(x, i), broadcast_value_at(y, i));
    });
    return result;
}

inline std::optional<int64_t> fixed_integer(const sym::expr& expression)
{
    const auto value = sym::fixed_value(expression);
    if(not value.has_value())
        return std::nullopt;
    return sym::to<int64_t>(*value);
}

inline std::optional<std::vector<int64_t>> fixed_integers(const symbolic_tensor_value& expressions)
{
    std::vector<int64_t> result;
    result.reserve(expressions.size());
    for(const auto& expression : expressions)
    {
        const auto value = fixed_integer(expression);
        if(not value.has_value())
            return std::nullopt;
        result.push_back(*value);
    }
    return result;
}

inline std::vector<shape::dynamic_dimension>
to_dynamic_dimensions(const symbolic_tensor_value& expressions)
{
    std::vector<shape::dynamic_dimension> result;
    result.reserve(expressions.size());
    transform(expressions, std::back_inserter(result), [](const auto& expression) {
        return shape::dynamic_dimension{expression};
    });
    return result;
}

inline std::vector<dim_like> to_reshape_dimensions(const symbolic_tensor_value& expressions)
{
    std::vector<dim_like> result;
    result.reserve(expressions.size());
    transform(expressions, std::back_inserter(result), [](const auto& expression) {
        const auto value = fixed_integer(expression);
        if(value.has_value())
            return dim_like{*value};
        return dim_like{shape::dynamic_dimension{expression}};
    });
    return result;
}

inline bool can_attach_symbolic_shape(const shape& input_shape)
{
    return not input_shape.dynamic() or input_shape.symbolic();
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
