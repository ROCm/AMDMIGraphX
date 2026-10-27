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
#include <migraphx/gpu/retune.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>
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
// and that retune cannot regenerate
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

std::string source_op_name(const code_object_op& co)
{
    return co.source.without_key().at("operator").at("op").at("name").to<std::string>();
}

bool is_ck_kernel(const code_object_op& co)
{
    return co.has_recipe() and starts_with(source_op_name(co), "gpu::ck_");
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

std::string solution_string(const value& v)
{
    auto s = v.without_key();
    if(s.is_null())
        return "-";
    if(s.is_string())
        return s.to<std::string>().empty() ? "-" : s.to<std::string>();
    return to_string(s);
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
        MIGRAPHX_THROW("retune: cannot retarget to " + arch + ": " + join_strings(errors, "; "));
}

} // namespace

void retune_ops::apply(program& p) const
{
    assert(ctx != nullptr);
    const auto from_ctx = options.contains("from") ? options.at("from").without_key() : value{};
    const bool force =
        options.contains("force") and options.at("force").without_key().to<bool>();
    if(not from_ctx.is_object() or not from_ctx.contains("compiled_for"))
        MIGRAPHX_THROW("retune: the program does not record the device it was compiled for; "
                       "recompile from the model");
    const auto from = from_value<device_description>(from_ctx.at("compiled_for").without_key());
    const auto to   = ctx->get_current_device().get_device_description();

    // Carry the streams and events of the schedule over to the new context
    std::size_t n_events  = 0;
    std::size_t n_streams = 1;
    if(from_ctx.contains("events"))
        n_events = from_ctx.at("events").without_key().to<std::size_t>();
    if(from_ctx.contains("streams"))
        n_streams = from_ctx.at("streams").without_key().to<std::size_t>();
    ctx->restore_schedule_state(n_events, n_streams);

    const bool same_target = from == to;
    if(same_target and not exhaustive_tune and not force)
    {
        log::info() << "retune: the program is already compiled for " << to.arch
                    << "; nothing to do";
        return;
    }

    std::vector<kernel_ref> kernels;
    std::vector<std::string> untouched;
    std::vector<std::string> unsupported;
    std::size_t with_recipe = 0;
    for(auto* mod : p.get_modules())
    {
        for(auto ins : iterator_for(*mod))
        {
            if(is_out_of_scope(ins->name()))
            {
                unsupported.push_back(describe(ins));
                continue;
            }
            if(ins->name() != "gpu::code_object")
                continue;
            const auto& co = any_cast<code_object_op>(ins->get_operator());
            if(co.has_recipe())
                with_recipe++;
            if(not co.has_recipe() or is_ck_kernel(co))
            {
                unsupported.push_back(describe(ins));
                continue;
            }
            kernels.push_back({mod, ins});
        }
    }

    const auto from_gfx = get_gfx_name(from.arch);
    const auto to_gfx   = get_gfx_name(to.arch);
    if(not same_target)
    {
        if(with_recipe == 0)
            MIGRAPHX_THROW("retune: the program was compiled without re-tune information; "
                           "recompile from the model with keep_recipes (--keep-recipes)");
        auto from_gen = get_gfx_generation(from.arch);
        auto to_gen   = get_gfx_generation(to.arch);
        if(to_gen < from_gen)
            MIGRAPHX_THROW("retune: MXR compiled for gfx" + std::to_string(from_gen) + " (" +
                           from_gfx + "); " + to_gfx + " is gfx" + std::to_string(to_gen) +
                           "; retargeting to an older generation is not supported");
        check_data_types(p, *ctx, to_gfx);
        if(not unsupported.empty())
            MIGRAPHX_THROW("retune: cannot retarget to " + to_gfx +
                           ": these instructions are compiled for " + from_gfx +
                           " and cannot be regenerated (CK, hipBLASLt / rocBLAS, MIOpen, or "
                           "kernels without a recipe): " +
                           join_names(unsupported));
    }

    // Select the kernels to recompile and the tuning level for each
    std::vector<kernel_ref> selected;
    std::vector<recompile_request> requests;
    for(const auto& k : kernels)
    {
        const auto& co      = any_cast<code_object_op>(k.ins->get_operator());
        const bool tunable  = not co.problem.is_null();
        const bool was_exh  = co.tuning_level == "exhaustive";
        const bool exh      = exhaustive_tune or was_exh;
        const bool at_level = was_exh == exh;
        if(same_target and not force and (at_level or not tunable))
        {
            untouched.push_back(describe(k.ins));
            continue;
        }
        selected.push_back(k);
        requests.push_back({k.ins, exh});
    }
    untouched.insert(untouched.end(), unsupported.begin(), unsupported.end());

    // Compile everything before changing the program, so a failure leaves it unchanged
    auto recompiled = recompile_kernels(*ctx, requests);

    std::stringstream report;
    for(auto i : range(selected.size()))
    {
        const auto& k   = selected[i];
        auto old        = any_cast<code_object_op>(k.ins->get_operator());
        const auto& nop = recompiled[i].code_object;
        const auto& nco = any_cast<code_object_op>(nop);
        k.mod->replace_instruction(k.ins, nop, k.ins->inputs(), k.ins->module_inputs());

        // Only tuned kernels are timed: time_op feeds random inputs, which can drive kernels
        // that take indices (gather, scatter) out of bounds and fault the whole device.
        const bool tunable = not nco.problem.is_null();
        std::string time   = tunable ? "not benchmarked" : "not timed";
        if(recompiled[i].benchmarked and tunable)
        {
            try
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << time_op(*ctx, nop, 1, 20) * 1000.0
                   << " us";
                time = ss.str();
            }
            catch(const std::exception& e)
            {
                time = std::string{"time unavailable: "} + e.what();
            }
        }
        report << "  " << nco.symbol_name << "  " << from_gfx << " -> " << to_gfx << "  "
               << solution_string(old.solution) << " -> " << solution_string(nco.solution) << "  "
               << (tunable ? nco.tuning_level : "-") << "  " << time;
        if(recompiled[i].excluded > 0)
            report << "  (" << recompiled[i].excluded << " signature-changing candidates excluded)";
        report << "\n";
    }
    log::info() << "retune: recompiled " << selected.size() << " kernels for " << to_gfx << "\n"
                << report.str();
    if(not untouched.empty())
        log::info() << "retune: left as compiled: " << join_names(untouched);
    if(get_gfx_generation(from.arch) != get_gfx_generation(to.arch))
        log::info() << "note: graph decisions were made for " << from_gfx
                    << "; recompile from the model for best performance on " << to_gfx << ".";
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
