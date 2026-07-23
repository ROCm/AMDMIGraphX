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
#include <migraphx/gpu/lower_device_ops.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/value.hpp>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

operation precompiled(instruction_ref ins)
{
    // gpu::contiguous appends a separate output allocation (additional_args == 1) and compiles as
    // the "contiguous" kernel; hip::fill/hip::copy already include their output buffer as an input
    if(ins->name() == "gpu::contiguous")
        return make_op("gpu::precompile_op",
                       {{"op", to_value(make_op("contiguous"))}, {"additional_args", 1}});
    return make_op("gpu::precompile_op",
                   {{"op", to_value(ins->get_operator())}, {"additional_args", 0}});
}

bool is_concat_op_name(const std::string& name)
{
    return name == "concat" or name == "fused_concat";
}

std::string embedded_op_name(const value& v)
{
    return v.contains("name") ? v.at("name").to<std::string>() : std::string{};
}

std::optional<value> get_concat_precompile_config(instruction_ref ins)
{
    if(not contains({"gpu::precompile_op", "gpu::dynamic_code_object_op"}, ins->name()))
        return std::nullopt;

    value v = ins->get_operator().to_value();
    std::optional<value> pre;
    if(ins->name() == "gpu::precompile_op")
        pre = v.contains("op") ? std::make_optional(v) : std::nullopt;
    else if(v.contains("pre_op") and v.at("pre_op").contains("operator"))
        pre = v.at("pre_op").at("operator");

    if(not pre or not pre->contains("op"))
        return std::nullopt;
    if(not is_concat_op_name(embedded_op_name(pre->at("op"))))
        return std::nullopt;
    return pre;
}

std::size_t precompile_output_args(const value& pre)
{
    return pre.contains("additional_args") ? pre.at("additional_args").to<std::size_t>() : 1;
}

bool needs_host_to_device_copy(instruction_ref ins)
{
    return contains({"@param", "@literal", "hip::copy_from_gpu"}, ins->name());
}

bool has_dynamic_concat_input(instruction_ref ins, const value& pre)
{
    const auto& inputs     = ins->inputs();
    const auto output_args = precompile_output_args(pre);
    if(inputs.size() <= output_args)
        return false;
    return std::any_of(inputs.begin(), inputs.end() - output_args, [](instruction_ref input) {
        return input->get_shape().dynamic();
    });
}

instruction_ref insert_gpu_copy(module& m, instruction_ref ins, instruction_ref input)
{
    auto alloc = m.insert_instruction(
        ins, make_op("hip::allocate", {{"shape", to_value(input->get_shape())}}));
    return m.insert_instruction(ins, make_op("hip::copy_to_gpu"), input, alloc);
}

void ensure_dynamic_concat_gpu_inputs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        auto pre = get_concat_precompile_config(ins);
        if(not pre or not has_dynamic_concat_input(ins, *pre))
            continue;

        const auto& inputs        = ins->inputs();
        const std::size_t ninputs = inputs.size() - precompile_output_args(*pre);

        std::vector<instruction_ref> new_inputs(inputs.begin(), inputs.end());
        bool changed = false;
        for(std::size_t i = 0; i < ninputs; ++i)
        {
            if(not needs_host_to_device_copy(inputs[i]))
                continue;
            new_inputs[i] = insert_gpu_copy(m, ins, inputs[i]);
            changed       = true;
        }
        if(changed)
            m.replace_instruction(ins, ins->get_operator(), new_inputs, ins->module_inputs());
    }
}

struct find_device_memory_op
{
    auto matcher() const { return match::name("hip::fill", "hip::copy", "gpu::contiguous"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(ins->get_shape().dynamic())
            return;
        m.replace_instruction(ins, precompiled(ins), ins->inputs(), ins->module_inputs());
    }
};

} // namespace

void lower_device_ops::apply(module& m) const
{
    ensure_dynamic_concat_gpu_inputs(m);
    match::find_matches(m, find_device_memory_op{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
