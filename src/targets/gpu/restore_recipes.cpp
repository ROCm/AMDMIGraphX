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
#include <migraphx/gpu/restore_recipes.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/record_compiled_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

struct kernel_ref
{
    module_ref mod;
    instruction_ref ins;
    /// The gpu::precompile_op in the recipe of ins
    instruction_ref recipe;
    std::vector<instruction_ref> inputs = {};
};

// Instructions whose solutions or binaries are valid only for the device they were compiled for
// and that recompile cannot regenerate
bool is_out_of_scope(const std::string& name)
{
    static const std::set<std::string> names = {"gpu::gemm",
                                                "gpu::quant_gemm",
                                                "gpu::hip_gemm",
                                                "gpu::hip_quant_gemm",
                                                "gpu::hipblaslt_op",
                                                "gpu::convolution",
                                                "gpu::quant_convolution",
                                                "gpu::convolution_backwards",
                                                "gpu::miopen_op",
                                                "gpu::miopen_fusion",
                                                "gpu::conv_bias",
                                                "gpu::conv_bias_relu",
                                                "gpu::lrn",
                                                "gpu::mlss_conv"};
    return contains(names, name);
}

std::string describe(instruction_ref ins)
{
    if(ins->name() == "gpu::code_object")
        return ins->name() + "[" + any_cast<code_object_op>(ins->get_operator()).symbol_name + "]";
    return ins->name();
}

std::string join_names(const std::vector<std::string>& names)
{
    const std::size_t max_names = 10;
    std::vector<std::string> shown(names.begin(),
                                   names.begin() + std::min(names.size(), max_names));
    auto result = join_strings(shown, ", ");
    if(names.size() > max_names)
        result += ", ... (" + std::to_string(names.size() - max_names) + " more)";
    return result;
}

struct kernel_scan
{
    std::vector<kernel_ref> kernels;
    std::vector<std::string> unsupported;
    bool any_recipe = false;
};

// The kernels that can be recompiled from their recipe, and the instructions that cannot
kernel_scan scan_kernels(const std::vector<module_ref>& mods)
{
    kernel_scan result;
    for(auto* mod : mods)
    {
        for(auto ins : iterator_for(*mod))
        {
            if(is_out_of_scope(ins->name()))
            {
                result.unsupported.push_back(describe(ins));
                continue;
            }
            if(ins->name() != "gpu::code_object")
                continue;
            auto recipe = find_recipe(ins);
            if(recipe.has_value())
                result.any_recipe = true;
            if(not recipe.has_value() or starts_with(recipe_op_name(*recipe), "gpu::ck_"))
            {
                result.unsupported.push_back(describe(ins));
                continue;
            }
            result.kernels.push_back({mod, ins, *recipe});
        }
    }
    return result;
}

void anchor_unused_instructions(module& m)
{
    if(m.begin() == m.end())
        return;
    auto last   = std::prev(m.end());
    auto unused = find_all(iterator_for(m), [&](instruction_ref ins) {
        return ins != last and ins->outputs().empty() and ins->get_shape().ndim() != 0 and
               not contains({"@param", "@literal", "@return"}, ins->name());
    });
    if(unused.empty())
        return;
    if(last->name() == "@return")
        m.insert_instruction(last, make_op("gpu::recompile_anchor"), unused);
    else
        m.add_instruction(make_op("gpu::recompile_anchor"), unused);
}

} // namespace

void restore_recipes::apply(module_pass_manager& mpm) const
{
    // Recompile from the root only, which covers every module
    if(&mpm.get_module() != mpm.get_root_module())
        return;
    assert(ctx != nullptr);
    auto* root = mpm.get_root_module();
    auto mods  = root->get_sub_modules();
    mods.insert(mods.begin(), root);

    const auto from_rd = get_compiled_for(*root);
    if(not from_rd.has_value())
        MIGRAPHX_THROW("recompile: the program does not record the device it was compiled for; "
                       "compile the model again with recompilable (--recompilable)");
    const auto& from    = *from_rd;
    const auto& to      = ctx->get_current_device().get_device_description();
    const auto from_gfx = get_gfx_name(from.arch);
    const auto to_gfx   = get_gfx_name(to.arch);
    const auto from_gen = get_gfx_generation(from.arch);
    const auto to_gen   = get_gfx_generation(to.arch);

    auto scan = scan_kernels(mods);
    if(from != to)
    {
        if(not scan.any_recipe)
            MIGRAPHX_THROW("recompile: the program was not compiled to be recompilable; "
                           "compile the model again with recompilable (--recompilable)");
        if(to_gen < from_gen)
            MIGRAPHX_THROW("recompile: MXR compiled for gfx" + std::to_string(from_gen) + " (" +
                           from_gfx + "); " + to_gfx + " is gfx" + std::to_string(to_gen) +
                           "; retargeting to an older generation is not supported");
        if(not scan.unsupported.empty())
            MIGRAPHX_THROW("recompile: cannot retarget to " + to_gfx +
                           ": these instructions are compiled for " + from_gfx +
                           " and cannot be regenerated (CK, hipBLASLt / rocBLAS, MIOpen, or "
                           "kernels without a recipe): " +
                           join_names(scan.unsupported));
    }

    // Find every input before changing the program, since inputs are found through other kernels
    for(auto& k : scan.kernels)
    {
        auto found = recipe_inputs(k.ins);
        if(not found.has_value())
            MIGRAPHX_THROW("recompile: a recipe parameter of " + describe(k.ins) +
                           " is not a path to its inputs; compile the model again with "
                           "recompilable (--recompilable)");
        k.inputs = std::move(*found);
    }

    for(auto* mod : mods)
    {
        if(not mod->bypass())
            anchor_unused_instructions(*mod);
    }
    for(const auto& k : scan.kernels)
        k.mod->replace_instruction(
            k.ins, k.recipe->get_operator(), k.inputs, k.recipe->module_inputs());

    log::info() << "recompile: recompiling " << scan.kernels.size() << " kernels for " << to_gfx
                << (exhaustive_tune ? " with exhaustive tuning" : "");
    if(not scan.unsupported.empty())
        log::info() << "recompile: left as compiled: " << join_names(scan.unsupported);
    if(from_gen != to_gen)
        log::info() << "note: graph decisions were made for " << from_gfx
                    << "; compile the model again for best performance on " << to_gfx << ".";
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
