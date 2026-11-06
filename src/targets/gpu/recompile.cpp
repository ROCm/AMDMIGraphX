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
#include <migraphx/gpu/recompile.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/record_compiled_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <iterator>
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

bool is_ck_kernel(instruction_ref recipe)
{
    return starts_with(recipe_op_name(recipe), "gpu::ck_");
}

std::string describe(instruction_ref ins)
{
    if(ins->name() == "gpu::code_object")
        return ins->name() + "[" + any_cast<code_object_op>(ins->get_operator()).symbol_name + "]";
    return ins->name();
}

std::string join_names(const std::vector<std::string>& names, std::size_t max_names = 10)
{
    std::vector<std::string> shown(names.begin(),
                                   names.begin() + std::min(names.size(), max_names));
    auto result = join_strings(shown, ", ");
    if(names.size() > max_names)
        result += ", ... (" + std::to_string(names.size() - max_names) + " more)";
    return result;
}

bool has_type(const shape& s, const std::set<shape::type_t>& types)
{
    if(s.type() == shape::tuple_type)
    {
        const auto& subs = s.sub_shapes();
        return std::any_of(
            subs.begin(), subs.end(), [&](const shape& sub) { return has_type(sub, types); });
    }
    return contains(types, s.type());
}

// Names of the instructions that produce or consume one of the types
std::vector<std::string> find_types(const program& p, const std::set<shape::type_t>& types)
{
    std::vector<std::string> found;
    for(const auto* mod : p.get_modules())
    {
        for(auto ins : iterator_for(*mod))
        {
            bool uses = has_type(ins->get_shape(), types) or
                        std::any_of(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
                            return has_type(i->get_shape(), types);
                        });
            if(uses and ins->name() != "@param" and ins->name() != "@literal" and
               ins->name() != "@return")
                found.push_back(mod->name() + ":" + describe(ins));
        }
    }
    return found;
}

void check_data_types(const program& p, const context& ctx, const std::string& arch)
{
    std::vector<std::string> errors;
    if(not gfx_has_fp8fnuz_intrinsics(ctx))
    {
        auto found = find_types(p, {shape::fp8e4m3fnuz_type, shape::fp8e5m2fnuz_type});
        if(not found.empty())
            errors.push_back("fp8 fnuz types, which " + arch +
                             " does not support, are used by: " + join_names(found));
    }
    if(not gfx_has_fp8ocp_intrinsics(ctx))
    {
        auto found = find_types(p, {shape::fp8e4m3fn_type, shape::fp8e5m2_type});
        if(not found.empty())
            errors.push_back("fp8 ocp types, which " + arch +
                             " does not support, are used by: " + join_names(found));
    }
    if(not errors.empty())
        MIGRAPHX_THROW("recompile: cannot retarget to " + arch + ": " + join_strings(errors, "; "));
}

struct kernel_scan
{
    std::vector<kernel_ref> kernels;
    std::vector<std::string> unsupported;
    std::size_t with_recipe = 0;
};

// The kernels that can be recompiled from their recipe, and the instructions that cannot
kernel_scan scan_kernels(program& p)
{
    kernel_scan result;
    for(auto* mod : p.get_modules())
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
                result.with_recipe++;
            if(not recipe.has_value() or is_ck_kernel(*recipe))
            {
                result.unsupported.push_back(describe(ins));
                continue;
            }
            result.kernels.push_back({mod, ins});
        }
    }
    return result;
}

void check_retarget(const program& p,
                    const context& ctx,
                    const kernel_scan& scan,
                    const device_description& from,
                    const device_description& to)
{
    const auto from_gfx = get_gfx_name(from.arch);
    const auto to_gfx   = get_gfx_name(to.arch);
    if(scan.with_recipe == 0)
        MIGRAPHX_THROW("recompile: the program was not compiled to be recompilable; "
                       "compile the model again with recompilable (--recompilable)");
    auto from_gen = get_gfx_generation(from.arch);
    auto to_gen   = get_gfx_generation(to.arch);
    if(to_gen < from_gen)
        MIGRAPHX_THROW("recompile: MXR compiled for gfx" + std::to_string(from_gen) + " (" +
                       from_gfx + "); " + to_gfx + " is gfx" + std::to_string(to_gen) +
                       "; retargeting to an older generation is not supported");
    check_data_types(p, ctx, to_gfx);
    if(not scan.unsupported.empty())
        MIGRAPHX_THROW("recompile: cannot retarget to " + to_gfx +
                       ": these instructions are compiled for " + from_gfx +
                       " and cannot be regenerated (CK, hipBLASLt / rocBLAS, MIOpen, or "
                       "kernels without a recipe): " +
                       join_names(scan.unsupported));
}

// Uses the instructions of a compiled module that nothing uses, so that the dead code
// elimination in compile_ops keeps them. A compiled module has such instructions on purpose: a
// kernel that writes its result into part of a buffer, such as an input of an eliminated
// concat, is not used by the instruction that reads the buffer.
struct recompile_anchor
{
    std::string name() const { return "gpu::recompile_anchor"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }
};

void anchor_unused_instructions(module& m)
{
    if(m.begin() == m.end())
        return;
    auto last = std::prev(m.end());
    std::vector<instruction_ref> unused;
    for(auto ins : iterator_for(m))
    {
        if(ins == last or not ins->outputs().empty() or ins->get_shape().ndim() == 0 or
           contains({"@param", "@literal", "@return"}, ins->name()))
            continue;
        unused.push_back(ins);
    }
    if(unused.empty())
        return;
    if(last->name() == "@return")
        m.insert_instruction(last, recompile_anchor{}, unused);
    else
        m.add_instruction(recompile_anchor{}, unused);
}

// The anchors, and the precompile ops that compile_ops replaced after its own dead code
// elimination, such as those of the fills it inserts for split-K
std::vector<instruction_ref> find_leftovers(module& m)
{
    std::vector<instruction_ref> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == recompile_anchor{}.name() or
           (ins->name() == "gpu::precompile_op" and ins->outputs().empty()))
            result.push_back(ins);
    }
    return result;
}

} // namespace

void restore_recipes::apply(program& p) const
{
    assert(ctx != nullptr);
    const auto* mm     = p.get_main_module();
    const auto from_rd = get_compiled_for(*mm);
    if(not from_rd.has_value())
        MIGRAPHX_THROW("recompile: the program does not record the device it was compiled for; "
                       "compile the model again with recompilable (--recompilable)");
    const auto& from = *from_rd;
    const auto to    = ctx->get_current_device().get_device_description();

    auto scan = scan_kernels(p);
    if(from != to)
        check_retarget(p, *ctx, scan, from, to);
    if(is_exhaustive(merge_compile_options(get_compile_options(*mm), options)))
        ctx->set_exhaustive_tune_flag(true);

    // Find every input before changing the program, since inputs are found through other kernels
    std::vector<std::vector<instruction_ref>> inputs;
    for(const auto& k : scan.kernels)
    {
        auto found = recipe_inputs(k.ins);
        if(not found.has_value())
            MIGRAPHX_THROW("recompile: the recipe of " + describe(k.ins) +
                           " is from an older build; compile the model again with "
                           "recompilable (--recompilable)");
        inputs.push_back(*found);
    }

    for(auto* mod : p.get_modules())
    {
        if(not mod->bypass())
            anchor_unused_instructions(*mod);
    }
    for(auto i : range(scan.kernels.size()))
    {
        const auto& k = scan.kernels[i];
        auto pre      = *find_recipe(k.ins);
        k.mod->replace_instruction(k.ins, pre->get_operator(), inputs[i], pre->module_inputs());
    }
    // compile_ops makes new recipe modules with the names of the old ones
    p.remove_unused_modules();

    const auto from_gfx = get_gfx_name(from.arch);
    const auto to_gfx   = get_gfx_name(to.arch);
    log::info() << "recompile: recompiling " << scan.kernels.size() << " kernels for " << to_gfx
                << (ctx->get_exhaustive_tune_flag() ? " with exhaustive tuning" : "");
    if(not scan.unsupported.empty())
        log::info() << "recompile: left as compiled: " << join_names(scan.unsupported);
    if(get_gfx_generation(from.arch) != get_gfx_generation(to.arch))
        log::info() << "note: graph decisions were made for " << from_gfx
                    << "; compile the model again for best performance on " << to_gfx << ".";
}

void remove_recompile_leftovers::apply(module& m) const
{
    for(auto ins : find_leftovers(m))
        m.remove_instruction(ins);
}

void check_recompiled_memory::apply(const module& m) const
{
    std::vector<std::string> users;
    for(const auto& ins : m)
    {
        if(ins.name() != "hip::allocate")
            continue;
        std::transform(ins.outputs().begin(),
                       ins.outputs().end(),
                       std::back_inserter(users),
                       [](instruction_ref out) { return describe(out); });
    }
    if(not users.empty())
        MIGRAPHX_THROW("recompile: the recompiled kernels need memory that the compiled program "
                       "does not have (used by " +
                       join_names(users) + "); compile the model again");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
