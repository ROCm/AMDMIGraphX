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
#ifndef MIGRAPHX_GUARD_OPERATORS_BIND_SYMBOLIC_HPP
#define MIGRAPHX_GUARD_OPERATORS_BIND_SYMBOLIC_HPP

#include <migraphx/config.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Operator used to bind a symbolic variable to input data. Such that
/// at runtime the symbolic variable map will be updated with the value
/// of the input. The operator only updates the symbolic variable map, it
/// is an identity operator otherwise. Used for data-dependent dimensions.
///
/// bind_symbolic(input) symbols = var("x"), var("y")
/// where input is tensor of shape [2].
/// at runtime set:
///     var("x") = input.at(0)
///     var("y") = input.at(1)
///
/// input must be a 1D static tensor and its dimension must match the number of symbols.
struct bind_symbolic
{
    std::vector<dim_like> symbols{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.symbols, "symbols"));
    }

    std::string name() const { return "bind_symbolic"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(1);
        if(symbols.size() != inputs.at(0).lens().at(0))
            MIGRAPHX_THROW("BIND_SYMBOLIC: dimension of input does not match number of symbols.");
        return inputs.at(0);
    }

    //TODO: have a function that tells how to link up the symbols to the inputs? Or keep it a simple 1 to 1?

    argument compute(shape, std::vector<argument> args) const
    {
        return args[0];
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
