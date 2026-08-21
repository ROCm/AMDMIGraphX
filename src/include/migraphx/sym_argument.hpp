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
#ifndef MIGRAPHX_GUARD_SYM_ARGUMENT_HPP
#define MIGRAPHX_GUARD_SYM_ARGUMENT_HPP

#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/tensor_view.hpp>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct sym_argument
{
    bool empty() const { return m_data.empty(); }

    const shape& get_shape() const { return m_shape; }

    tensor_view<sym::expr> get()
    {
        if(empty())
            return {};
        return make_view(m_shape, m_data.data());
    }

    tensor_view<const sym::expr> get() const
    {
        if(empty())
            return {};
        return make_view(m_shape, m_data.data());
    }

    friend bool operator==(const sym_argument& x, const sym_argument& y)
    {
        if(x.get_shape() != y.get_shape() or x.empty() != y.empty())
            return false;
        return x.empty() or x.get() == y.get();
    }

    friend bool operator!=(const sym_argument& x, const sym_argument& y) { return not(x == y); }

    std::vector<sym::expr> m_data;
    shape m_shape;
};

inline bool is_sym_argument_type(shape::type_t type)
{
    return type != shape::tuple_type and shape::is_computable(type);
}

inline sym_argument allocate_sym_argument(const shape& s)
{
    return {std::vector<sym::expr>(s.element_space()), s};
}

inline std::optional<int64_t> fixed_integer(const sym::expr& expression)
{
    const auto value = sym::fixed_value(expression);
    if(not value.has_value())
        return std::nullopt;
    return sym::to<int64_t>(*value);
}

inline std::optional<std::vector<int64_t>> fixed_integers(const sym_argument& argument)
{
    if(argument.empty())
        return std::nullopt;
    std::vector<int64_t> result;
    result.reserve(argument.get_shape().elements());
    for(const auto& expression : argument.get())
    {
        const auto value = fixed_integer(expression);
        if(not value)
            return std::nullopt;
        result.push_back(*value);
    }
    return result;
}

inline bool sym_argument_matches_shape(const shape& output_shape, const sym_argument& argument)
{
    if(argument.empty() or argument.get_shape() != output_shape or
       not is_sym_argument_type(output_shape.type()) or output_shape.dynamic() or
       argument.m_data.size() < output_shape.element_space())
        return false;
    return none_of(argument.get(), [](const auto& expression) { return expression.empty(); });
}

template <class F>
sym_argument
compute_sym_unary(const shape& output_shape, const std::vector<sym_argument>& args, F f)
{
    if(args.size() != 1 or not sym_argument_matches_shape(args[0].get_shape(), args[0]) or
       output_shape.dynamic() or not is_sym_argument_type(output_shape.type()) or
       args[0].get_shape().lens() != output_shape.lens())
        return {};

    auto result      = allocate_sym_argument(output_shape);
    const auto input = args[0].get();
    auto output      = result.get();
    std::transform(input.begin(), input.end(), output.begin(), std::move(f));
    if(not sym_argument_matches_shape(output_shape, result))
        return {};
    return result;
}

template <class F>
sym_argument
compute_sym_binary(const shape& output_shape, const std::vector<sym_argument>& args, F f)
{
    if(args.size() != 2 or not sym_argument_matches_shape(args[0].get_shape(), args[0]) or
       not sym_argument_matches_shape(args[1].get_shape(), args[1]) or output_shape.dynamic() or
       not is_sym_argument_type(output_shape.type()) or
       args[0].get_shape().lens() != output_shape.lens() or
       args[1].get_shape().lens() != output_shape.lens())
        return {};

    auto result  = allocate_sym_argument(output_shape);
    const auto x = args[0].get();
    const auto y = args[1].get();
    auto output  = result.get();
    std::transform(x.begin(), x.end(), y.begin(), output.begin(), std::move(f));
    if(not sym_argument_matches_shape(output_shape, result))
        return {};
    return result;
}

inline sym_argument pass_through_sym_argument(const shape& output_shape,
                                              const std::vector<sym_argument>& args,
                                              std::size_t input_index = 0)
{
    if(input_index >= args.size())
        return {};
    const auto& input = args[input_index];
    if(not sym_argument_matches_shape(input.get_shape(), input) or output_shape.dynamic() or
       not is_sym_argument_type(output_shape.type()) or
       input.get_shape().elements() != output_shape.elements())
        return {};

    auto result       = allocate_sym_argument(output_shape);
    const auto values = input.get();
    auto output       = result.get();
    std::copy(values.begin(), values.end(), output.begin());
    if(not sym_argument_matches_shape(output_shape, result))
        return {};
    return result;
}

inline sym_argument broadcast_sym_argument(const shape& output_shape,
                                           const std::vector<sym_argument>& args,
                                           std::size_t input_index = 0)
{
    if(input_index >= args.size())
        return {};
    const auto& input = args[input_index];
    if(not sym_argument_matches_shape(input.get_shape(), input) or output_shape.dynamic() or
       not is_sym_argument_type(output_shape.type()) or
       input.m_data.size() < output_shape.element_space())
        return {};

    auto result    = input;
    result.m_shape = output_shape;
    if(not sym_argument_matches_shape(output_shape, result))
        return {};
    return result;
}

inline std::vector<shape::dynamic_dimension> to_dynamic_dimensions(const sym_argument& argument)
{
    std::vector<shape::dynamic_dimension> result;
    result.reserve(argument.get_shape().elements());
    transform(argument.get(), std::back_inserter(result), [](const auto& expression) {
        return shape::dynamic_dimension{expression};
    });
    return result;
}

inline std::vector<dim_like> to_reshape_dimensions(const sym_argument& argument)
{
    std::vector<dim_like> result;
    result.reserve(argument.get_shape().elements());
    transform(argument.get(), std::back_inserter(result), [](const auto& expression) {
        const auto value = fixed_integer(expression);
        if(value.has_value())
            return dim_like{*value};
        return dim_like{shape::dynamic_dimension{expression}};
    });
    return result;
}

inline bool is_static_or_symbolic_shape(const shape& input_shape)
{
    return not input_shape.dynamic() or input_shape.symbolic();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
