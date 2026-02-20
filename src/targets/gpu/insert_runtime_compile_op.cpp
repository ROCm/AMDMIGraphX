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
#include <migraphx/gpu/insert_runtime_compile_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void insert_runtime_compile_op::apply(module& m) const
{
    // Check if there are any dynamic_code_object_op instructions (short-circuit on first match)
    auto has_dynamic_op = std::any_of(iterator_for(m).begin(), iterator_for(m).end(), [](auto ins) {
        return ins->name() == "gpu::dynamic_code_object_op";
    });

    // If there are no dynamic instructions, nothing to do
    if(not has_dynamic_op)
        return;

    // Get module parameters and move them all to the beginning
    auto params = m.get_parameter_names();
    std::vector<instruction_ref> param_ins;
    for(const auto& name : params)
    {
        param_ins.push_back(m.get_parameter(name));
    }

    // Move all parameters to the beginning in order
    // Find the first param instruction in the module
    auto insert_pos = *std::find_if(iterator_for(m).begin(), iterator_for(m).end(), [](auto ins) {
        return ins->name() == "@param";
    });

    for(auto param_ref : param_ins)
    {
        m.move_instruction(param_ref, insert_pos);
        insert_pos = std::next(param_ref);
    }

    // Create runtime_compile_op operation
    auto rt_compile_op = make_op("gpu::runtime_compile_op");

    // Insert the runtime_compile_op after all parameters
    // It takes the module parameters as inputs and the module itself as a module arg
    m.insert_instruction(insert_pos, rt_compile_op, param_ins, {&m});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
