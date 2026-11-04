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
#include <migraphx/serialize.hpp>
#include <array>
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

// The widest of 4, 2, 1 that divides n while (n / width) * other_tiles still
// covers the threads; 1 always qualifies
static std::size_t pick_tile(std::size_t n, std::size_t other_tiles, std::size_t threads)
{
    const std::array<std::size_t, 3> widths{4, 2, 1};
    return *std::find_if(widths.begin(), widths.end(), [&](auto t) {
        return t == 1 or (n % t == 0 and (n / t) * other_tiles >= threads);
    });
}

static std::string shape_text(const shape& s)
{
    return "make_shape(index_ints<" + to_string_range(s.lens()) + ">{}, index_ints<" +
           to_string_range(s.strides()) + ">{})";
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

        const auto state      = from_value<shape>(v.at("state_shape"));
        const auto chunks     = v.at("chunks").to<std::size_t>();
        const bool has_state  = v.at("has_state").to<bool>();
        const bool update_dot = v.at("has_update_dot").to<bool>();
        const bool has_read   = v.at("has_read").to<bool>();
        const bool has_gamma  = v.at("has_gamma").to<bool>();
        const bool has_out    = v.at("has_out").to<bool>();
        auto views            = from_value<std::vector<shape>>(v.at("views"));
        auto offsets          = v.at("offsets").to_vector<std::size_t>();
        auto deltas           = v.at("deltas").to_vector<std::size_t>();

        // the families present, in the op's input order
        std::vector<std::string> present;
        if(has_read)
            present.push_back("w");
        present.push_back("q");
        if(update_dot)
        {
            present.push_back("kt");
            present.push_back("u0");
        }
        else
        {
            present.push_back("upd");
        }
        if(has_gamma)
            present.push_back("gamma");
        if(has_out)
            present.push_back("pu0");

        const auto dk   = state.lens().at(state.ndim() - 2);
        const auto dv   = state.lens().back();
        const auto q_i  = std::find(present.begin(), present.end(), "q") - present.begin();
        const auto rows = views.at(q_i).lens().at(views.at(q_i).ndim() - 2);
        // the widest dv block that divides dv, up to 32 columns
        std::size_t dvb = 1;
        for(std::size_t b = 1; b <= std::min<std::size_t>(32, dv); b++)
            if(dv % b == 0)
                dvb = b;
        const std::size_t tv  = pick_tile(dvb, 1, 1);
        const std::size_t nvt = dvb / tv;
        const std::size_t tr  = pick_tile(rows, nvt, nthreads);
        const std::size_t td  = pick_tile(dk, nvt, nthreads);

        std::string settings;
        settings += "static constexpr index_int chunks = " + std::to_string(chunks) + ";\n";
        settings += "static constexpr index_int rows = " + std::to_string(rows) + ";\n";
        settings += "static constexpr index_int dk = " + std::to_string(dk) + ";\n";
        settings += "static constexpr index_int dv = " + std::to_string(dv) + ";\n";
        settings += "static constexpr index_int dvb = " + std::to_string(dvb) + ";\n";
        settings += "static constexpr index_int tv = " + std::to_string(tv) + ";\n";
        settings += "static constexpr index_int tr = " + std::to_string(tr) + ";\n";
        settings += "static constexpr index_int td = " + std::to_string(td) + ";\n";
        for(const auto& [flag, on] :
            std::vector<std::pair<std::string, bool>>{{"has_state", has_state},
                                                      {"has_update_dot", update_dot},
                                                      {"has_read", has_read},
                                                      {"has_gamma", has_gamma},
                                                      {"has_out", has_out}})
            settings += "static constexpr bool " + flag + " = " + (on ? "true" : "false") + ";\n";
        // kernel arguments: [state0], families..., output, state_out
        std::vector<std::string> arg_names;
        if(has_state)
            arg_names.push_back("s0");
        arg_names.insert(arg_names.end(), present.begin(), present.end());
        arg_names.push_back("output");
        arg_names.push_back("state_out");
        // absent roles alias q, which the kernel never reads through them
        auto arg_index = [&](const std::string& role) {
            auto it = std::find(arg_names.begin(), arg_names.end(), role);
            if(it == arg_names.end())
                it = std::find(arg_names.begin(), arg_names.end(), "q");
            return it - arg_names.begin();
        };
        for(const auto& role : std::vector<std::string>{
                "s0", "w", "q", "kt", "u0", "upd", "gamma", "pu0", "output", "state_out"})
            settings += "static constexpr index_int " + role +
                        "_index = " + std::to_string(arg_index(role)) + ";\n";
        std::vector<std::string> families{"w", "q", "kt", "u0", "upd", "gamma", "pu0"};
        for(const auto& f : families)
        {
            auto it = std::find(present.begin(), present.end(), f);
            auto k  = (it == present.end() ? q_i : it - present.begin());
            settings +=
                "static constexpr auto " + f + "_shape = " + shape_text(views.at(k)) + ";\n";
            settings += "static constexpr index_int " + f +
                        "_offset = " + std::to_string(offsets.at(k)) + ";\n";
            settings += "static constexpr index_int " + f +
                        "_delta = " + std::to_string(deltas.at(k)) + ";\n";
        }

        auto problems = state.elements() / (dk * dv);
        options.set_launch_params(v, problems * (dv / dvb) * nthreads, nthreads);
        auto src = interpolate_string(chunk_scan_kernel,
                                      {{"preamble", v.at("preamble").to<std::string>()},
                                       {"settings", settings},
                                       {"params", enum_params(arg_names.size(), "void * p")},
                                       {"args", enum_params(arg_names.size(), "p")}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v    = op.to_value();
        auto mods = ins->module_inputs();
        std::string preamble;
        std::size_t k = 0;
        preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_step");
        if(v.at("has_read").to<bool>())
            preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_read");
        else
            preamble += stub_pointwise("chunk_scan_read");
        if(v.at("has_out").to<bool>())
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
