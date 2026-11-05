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

#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const chunk_scan_kernel = R"__migraphx__(
#include <migraphx/kernels/chunk_scan.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

MIGRAPHX_LIFT_CLASS(chunk_scan_step_t, chunk_scan_step);
MIGRAPHX_LIFT_CLASS(chunk_scan_read_t, chunk_scan_read);
MIGRAPHX_LIFT_CLASS(chunk_scan_out_t, chunk_scan_out);

struct chunk_scan_settings
{
${settings}
};

extern "C" {

MIGRAPHX_GLOBAL void chunk_scan_kernel(${params})
{
    make_tensors()(${args})([](auto... xs) {
        chunk_scan<chunk_scan_settings>(
            chunk_scan_step_t{}, chunk_scan_read_t{}, chunk_scan_out_t{}, xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

// A pointwise hole the op does not use: returns its first argument
static std::string stub_pointwise(const std::string& name)
{
    return "template <class T, class... Ts>\n__device__ T " + name + "(T x, Ts...) { return x; }\n";
}

struct chunk_scan_compiler : compiler<chunk_scan_compiler>
{
    std::vector<std::string> names() const { return {"gpu::chunk_scan"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        constexpr std::size_t nthreads = 256;
        hip_compile_options options;
        options.output      = inputs.back();
        options.inputs      = flatten_tuple_shapes(inputs);
        options.kernel_name = "chunk_scan_kernel";

        const auto state     = from_value<shape>(v.at("state_shape"));
        const auto chunks    = v.at("chunks").to<std::size_t>();
        const bool has_state = v.at("has_state").to<bool>();
        const auto families  = v.at("families").to_vector<std::string>();
        const auto views     = from_value<std::vector<shape>>(v.at("views"));
        const auto offsets   = v.at("offsets").to_vector<std::size_t>();
        const auto deltas    = v.at("deltas").to_vector<std::size_t>();
        auto has             = [&](const std::string& role) { return contains(families, role); };

        const auto dk   = state.lens().at(state.ndim() - 2);
        const auto dv   = state.lens().back();
        const auto q_i  = std::find(families.begin(), families.end(), "q") - families.begin();
        const auto rows = views.at(q_i).lens().at(views.at(q_i).ndim() - 2);
        // the widest dv block up to 32 columns that divides dv and keeps the
        // state and u tiles, accumulated in float, in LDS
        constexpr std::size_t max_lds = 64 * 1024;
        const std::size_t acc_size    = state.type() == shape::double_type ? 8 : 4;
        std::size_t dvb               = 0;
        for(std::size_t b = 1; b <= std::min<std::size_t>(32, dv); b++)
        {
            if(dv % b == 0 and (dk + rows) * b * acc_size <= max_lds)
                dvb = b;
        }
        if(dvb == 0)
            MIGRAPHX_THROW("chunk_scan: the state tile does not fit in LDS");

        std::string settings;
        settings += "static constexpr index_int chunks = " + std::to_string(chunks) + ";\n";
        settings += "static constexpr index_int rows = " + std::to_string(rows) + ";\n";
        settings += "static constexpr index_int dk = " + std::to_string(dk) + ";\n";
        settings += "static constexpr index_int dv = " + std::to_string(dv) + ";\n";
        settings += "static constexpr index_int dvb = " + std::to_string(dvb) + ";\n";
        for(const auto& [flag, on] :
            std::vector<std::pair<std::string, bool>>{{"has_state", has_state},
                                                      {"has_update_dot", has("kt")},
                                                      {"has_read", has("w")},
                                                      {"has_gamma", has("gamma")},
                                                      {"has_out", has("pu0")}})
            settings += "static constexpr bool " + flag + " = " + (on ? "true" : "false") + ";\n";
        // kernel arguments: [state0], families..., output, state_out
        const std::size_t nargs = (has_state ? 1 : 0) + families.size() + 2;
        settings += "static constexpr index_int s0_index = 0;\n";
        settings +=
            "static constexpr index_int output_index = " + std::to_string(nargs - 2) + ";\n";
        settings +=
            "static constexpr index_int state_out_index = " + std::to_string(nargs - 1) + ";\n";
        // each family: its argument and its chunk 0 view, offset and stride
        for(std::size_t k = 0; k < families.size(); k++)
        {
            settings += "struct " + families[k] + " {\n";
            settings +=
                "static constexpr index_int index = " + std::to_string(k + (has_state ? 1 : 0)) +
                ";\n";
            settings += "static constexpr auto view = " + generate_make_shape(views[k]) + ";\n";
            settings += "static constexpr index_int offset = " + std::to_string(offsets[k]) + ";\n";
            settings += "static constexpr index_int delta = " + std::to_string(deltas[k]) + ";\n";
            settings += "};\n";
        }
        // absent roles alias q; the kernel never reads through them
        for(const auto& role : std::vector<std::string>{"w", "kt", "u0", "upd", "gamma", "pu0"})
        {
            if(not has(role))
                settings += "using " + role + " = q;\n";
        }

        auto problems = state.elements() / (dk * dv);
        options.set_launch_params(v, problems * (dv / dvb) * nthreads, nthreads);
        auto src = interpolate_string(chunk_scan_kernel,
                                      {{"preamble", v.at("preamble").to<std::string>()},
                                       {"settings", settings},
                                       {"params", enum_params(nargs, "void * p")},
                                       {"args", enum_params(nargs, "p")}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v              = op.to_value();
        const auto& mods    = ins->module_inputs();
        const auto families = v.at("families").to_vector<std::string>();
        std::string preamble;
        std::size_t k = 0;
        preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_step");
        if(contains(families, "w"))
            preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_read");
        else
            preamble += stub_pointwise("chunk_scan_read");
        if(contains(families, "pu0"))
            preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_out");
        else
            preamble += stub_pointwise("chunk_scan_out");
        v["preamble"] = preamble;
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
