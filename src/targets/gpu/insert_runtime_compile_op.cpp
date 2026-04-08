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
#include <migraphx/pass_manager.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void insert_runtime_compile_op::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();
    if(*mpm.get_root_module() != m)
        return;

    auto has_dynamic_op = std::any_of(iterator_for(m).begin(), iterator_for(m).end(), [](auto ins) {
        return ins->name() == "gpu::dynamic_code_object_op";
    });

    if(not has_dynamic_op)
        return;

    auto params = m.get_parameter_names();
    std::sort(params.begin(), params.end());
    std::vector<instruction_ref> param_ins(params.size());
    std::transform(params.begin(), params.end(), param_ins.begin(), [&](const auto& name) {
        return m.get_parameter(name);
    });

    module* submod = mpm.create_module("runtime_compile: " + m.name());
    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(const auto& name : params)
        ins_map[m.get_parameter(name)] = submod->add_parameter(name, m.get_parameter_shape(name));

    auto submod_outputs = submod->insert_instructions(submod->end(), m.begin(), m.end(), &ins_map);
    submod->add_return(submod_outputs);

    auto return_ins = std::prev(m.end());
    auto rt_compile_ins =
        m.insert_instruction(return_ins, make_op("gpu::runtime_compile_op"), param_ins, {submod});

    std::vector<instruction_ref> return_args;
    if(rt_compile_ins->get_shape().type() == shape::tuple_type)
    {
        const auto& sub_shapes = rt_compile_ins->get_shape().sub_shapes();
        return_args.reserve(sub_shapes.size());
        for(std::size_t i = 0; i < sub_shapes.size(); ++i)
            return_args.push_back(m.insert_instruction(
                return_ins, make_op("get_tuple_elem", {{"index", i}}), {rt_compile_ins}));
    }
    else
    {
        return_args = {rt_compile_ins};
    }
    m.replace_return(return_args);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
