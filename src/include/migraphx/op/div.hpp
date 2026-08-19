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
#ifndef MIGRAPHX_GUARD_OPERATORS_DIV_HPP
#define MIGRAPHX_GUARD_OPERATORS_DIV_HPP

#include <migraphx/config.hpp>
#include <migraphx/op/binary.hpp>
#include <migraphx/symbolic_tensor_value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct div : binary<div>
{
    std::string point_function() const { return "/"; }
    auto apply() const
    {
        return [](auto x, auto y) { return x / y; };
    }
    std::optional<symbolic_tensor_value>
    symbolic_compute(const shape& output_shape,
                     const std::vector<shape>&,
                     const std::vector<std::optional<symbolic_tensor_value>>& input_values) const
    {
        if(output_shape.type() != shape::int64_type)
            return std::nullopt;
        return compute_symbolic_binary(
            output_shape, input_values, [](const auto& x, const auto& y) {
                if(y.eval_interval().contains(0))
                    return sym::expr{};
                return x / y;
            });
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
