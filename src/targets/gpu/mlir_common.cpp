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

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <cassert>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// rocMLIR can only map a layout with a unit stride to memory
static bool has_unit_stride(const shape& s) { return s.standard() or contains(s.strides(), 1); }

static shape append_unit_dim(const shape& s)
{
    auto lens    = s.lens();
    auto strides = s.strides();
    lens.push_back(1);
    strides.push_back(1);
    return {s.type(), lens, strides};
}

// Unsqueeze the return values whose output layout has no unit stride
static std::vector<shape> adjust_return_shapes(module& m, const std::vector<shape>& outputs)
{
    auto ret = std::prev(m.end());
    assert(ret->name() == "@return");
    auto returns = ret->inputs();
    assert(returns.size() == outputs.size());
    std::vector<instruction_ref> new_returns;
    std::transform(returns.begin(),
                   returns.end(),
                   outputs.begin(),
                   std::back_inserter(new_returns),
                   [&](instruction_ref ins, const shape& s) {
                       if(has_unit_stride(s))
                           return ins;
                       return m.insert_instruction(
                           ret, make_op("unsqueeze", {{"axes", {s.ndim()}}}), ins);
                   });
    if(new_returns != returns)
        m.replace_return(new_returns);
    std::vector<shape> result;
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(result), [](const shape& s) {
        return has_unit_stride(s) ? s : append_unit_dim(s);
    });
    return result;
}

std::vector<shape> adjust_param_shapes(module& m, const std::vector<shape>& inputs)
{
    auto result = inputs;
    auto names  = m.get_parameter_names();
    std::sort(names.begin(), names.end());
    for(auto i : range(names.size()))
    {
        const auto& name  = names[i];
        const auto& input = inputs[i];
        auto param        = m.get_parameter(name);
        assert(param->get_shape().standard());
        if(input.standard())
            continue;
        instruction_ref new_param;
        if(has_unit_stride(input))
        {
            new_param = m.add_parameter(name + ".0", input);
        }
        else
        {
            // Give the buffer a trailing unit dimension and squeeze it away
            // inside the kernel so the layout stays expressible
            auto unit_param = m.add_parameter(name + ".0", append_unit_dim(input));
            new_param       = m.insert_instruction(
                std::next(unit_param), make_op("squeeze", {{"axes", {input.ndim()}}}), unit_param);
        }
        m.replace_instruction(param, new_param);
        m.remove_instruction(param);
    }
    // The output buffers are handled the same way with an unsqueeze before the return
    const auto& output = inputs.back();
    if(output.type() == shape::tuple_type)
        result.back() = shape{adjust_return_shapes(m, output.sub_shapes())};
    else
        result.back() = adjust_return_shapes(m, {output}).front();
    return result;
}

instruction_ref insert_mlir(module& m,
                            instruction_ref ins,
                            code_object_op co,
                            const std::vector<instruction_ref>& inputs)
{
    auto refs          = inputs;
    co.expected_inputs = to_shapes(refs);
    co.output_arg      = refs.size() - 1;
    return m.insert_instruction(ins, co, refs);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
