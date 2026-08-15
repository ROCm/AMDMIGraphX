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
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/symbolic_tensor_value.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <iterator>
#include <limits>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

std::optional<symbolic_tensor_value>
onnx_parser::get_symbolic_tensor_value(instruction_ref ins) const
{
    if(const auto found = symbolic_tensor_values.find(ins); found != symbolic_tensor_values.end())
        return found->second;
    if(not shape::is_integral(ins->get_shape().type()))
        return std::nullopt;

    const auto value = ins->eval();
    if(value.empty())
        return std::nullopt;

    symbolic_tensor_value result;
    bool converted = false;
    value.visit([&](auto input) {
        using type = std::remove_cv_t<typename decltype(input)::value_type>;
        if constexpr(std::is_integral<type>{})
        {
            if constexpr(std::is_unsigned<type>{} and sizeof(type) >= sizeof(int64_t))
            {
                if(any_of(input, [](auto x) {
                       return x > static_cast<type>(std::numeric_limits<int64_t>::max());
                   }))
                    return;
            }
            transform(input, std::back_inserter(result), [](auto x) { return sym::lit(x); });
            converted = true;
        }
    });
    if(not converted)
        return std::nullopt;
    return result;
}

bool symbolic_propagate_context::output_has_elements(std::size_t count,
                                                     std::size_t output_index) const
{
    if(output_index >= results.size())
        return false;
    const auto& output_shape = results[output_index]->get_shape();
    return not output_shape.dynamic() and output_shape.elements() == count;
}

void symbolic_propagate_context::pass_through(std::size_t input_index,
                                              std::size_t output_index) const
{
    auto value = arg(input_index);
    if(not value.has_value() or output_index >= results.size())
        return;
    const auto& output_shape = results[output_index]->get_shape();
    if(not shape::is_integral(output_shape.type()) or
       not output_has_elements(value->size(), output_index))
        return;
    set(std::move(*value), output_index);
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
