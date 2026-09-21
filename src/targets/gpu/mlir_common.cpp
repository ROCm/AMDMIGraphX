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
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void adjust_param_shapes(module& m, const std::vector<shape>& inputs)
{
    auto names = m.get_parameter_names();
    std::sort(names.begin(), names.end());
    for(auto i : range(names.size()))
    {
        const auto& name  = names[i];
        const auto& input = inputs[i];
        auto param        = m.get_parameter(name);
        assert(param->get_shape().standard());
        if(input.standard())
            continue;
        auto new_param = m.add_parameter(name + ".0", input);
        m.replace_instruction(param, new_param);
        m.remove_instruction(param);
    }
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
