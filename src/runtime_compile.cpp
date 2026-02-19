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
#include <migraphx/runtime_compile.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct runtime_compile_op
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "runtime_compile_op"; }

    shape compute_shape(const std::vector<shape>&, const std::vector<module_ref>&) const
    {
        return {};
    }
};
MIGRAPHX_REGISTER_OP(runtime_compile_op);

void find_dyn_ins_shapes(
    module_ref target_mod,
    std::unordered_map<std::string, shape> param_shapes,
    std::unordered_map<instruction_ref, std::vector<shape>>& compile_input_shapes)
{
    std::unordered_map<instruction_ref, shape> ins_shapes;
    for(auto ins : iterator_for(*target_mod))
    {
        if(ins->name() == "@param")
        {
            auto param_name = any_cast<builtin::param>(ins->get_operator()).parameter;
            ins_shapes[ins] = param_shapes.at(param_name);
        }
        else if(ins->name() == "@literal")
        {
            ins_shapes[ins] = ins->get_shape();
        }
        else
        {
            std::vector<shape> input_shapes;
            input_shapes.reserve(ins->inputs().size());
            for(auto input_ins : ins->inputs())
            {
                input_shapes.push_back(ins_shapes.at(input_ins));
            }
            if(ins->name() == "gpu::dynamic_code_object_op")
            {
                compile_input_shapes[ins] = input_shapes;
            }
            ins_shapes[ins] = ins->get_operator().compute_shape(input_shapes, ins->module_inputs());
        }
    }
}

void compile_dyn_ins(context& ctx,
                     module_ref& mod,
                     const std::unordered_map<std::string, shape>& param_shapes)
{
    std::unordered_map<instruction_ref, std::vector<shape>> compile_input_shapes;
    find_dyn_ins_shapes(mod, param_shapes, compile_input_shapes);

    if(compile_input_shapes.empty())
        return;

    std::vector<instruction_ref> runtime_dyn_inss;
    runtime_dyn_inss.reserve(compile_input_shapes.size());
    std::transform(compile_input_shapes.begin(),
                   compile_input_shapes.end(),
                   std::back_inserter(runtime_dyn_inss),
                   [](const auto& kv) { return kv.first; });

    par_for(runtime_dyn_inss.size(), [&](std::size_t i) {
        auto ins = runtime_dyn_inss[i];
        assert(ins->get_operator().name() == "gpu::dynamic_code_object_op");

        ins->get_operator().runtime_compile(
            ctx, compile_input_shapes.at(ins), ins->module_inputs());
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
