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
#include <cassert>

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
    static constexpr index_int chunks          = ${chunks};
    static constexpr index_int rows            = ${rows};
    static constexpr index_int dk              = ${dk};
    static constexpr index_int dv              = ${dv};
    static constexpr index_int dvb             = ${dvb};
    static constexpr index_int max_lds         = ${max_lds};
    static constexpr bool has_state            = ${has_state};
    static constexpr bool has_update_dot       = ${has_update_dot};
    static constexpr bool has_read             = ${has_read};
    static constexpr bool has_decay            = ${has_decay};
    static constexpr bool has_out              = ${has_out};
    static constexpr index_int s0_index        = 0;
    static constexpr index_int output_index    = ${output_index};
    static constexpr index_int state_out_index = ${state_out_index};
${families}
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
        assert(contains(families, "output_matrix"));
        // kernel arguments: [state0], families..., output, state_out
        const std::size_t first_family = has_state ? 1 : 0;
        const std::size_t nargs        = first_family + families.size() + 2;

        const auto dk = state.lens().at(state.ndim() - 2);
        const auto dv = state.lens().back();
        const auto matrix_index =
            std::find(families.begin(), families.end(), "output_matrix") - families.begin();
        const auto& output_matrix = inputs.at(first_family + matrix_index);
        const auto rows           = output_matrix.lens().at(output_matrix.ndim() - 2);
        // the widest block of up to 32 columns that divides dv and keeps the
        // state and u tiles, in the accumulation type, in LDS
        const auto max_lds         = ctx.get_current_device().get_max_shared_memory_per_block();
        const std::size_t acc_size = output_matrix.type() == shape::double_type ? 8 : 4;
        const auto blocks =
            divisors_up_to(dv, std::min<std::size_t>(32, max_lds / ((dk + rows) * acc_size)));
        if(blocks.empty())
            MIGRAPHX_THROW("chunk_scan: the state tile does not fit in LDS");
        const auto dvb = blocks.back();

        // each family names its argument
        std::string family_args;
        for(std::size_t k = 0; k < families.size(); k++)
            family_args += "struct " + families[k] + " { static constexpr index_int index = " +
                           std::to_string(first_family + k) + "; };\n";
        // absent roles alias the output matrix; the kernel never reads through them
        for(const auto& role : std::vector<std::string>{"read_matrix",
                                                        "update_matrix",
                                                        "update_input",
                                                        "state_update",
                                                        "decay",
                                                        "output_base"})
        {
            if(not contains(families, role))
                family_args += "using " + role + " = output_matrix;\n";
        }

        auto problems = state.elements() / (dk * dv);
        options.set_launch_params(v, problems * (dv / dvb) * nthreads, nthreads);
        auto src = interpolate_string(
            chunk_scan_kernel,
            {{"preamble", v.at("preamble").to<std::string>()},
             {"chunks", std::to_string(chunks)},
             {"rows", std::to_string(rows)},
             {"dk", std::to_string(dk)},
             {"dv", std::to_string(dv)},
             {"dvb", std::to_string(dvb)},
             {"max_lds", std::to_string(max_lds)},
             {"has_state", has_state ? "true" : "false"},
             {"has_update_dot", contains(families, "update_matrix") ? "true" : "false"},
             {"has_read", contains(families, "read_matrix") ? "true" : "false"},
             {"has_decay", contains(families, "decay") ? "true" : "false"},
             {"has_out", contains(families, "output_base") ? "true" : "false"},
             {"output_index", std::to_string(nargs - 2)},
             {"state_out_index", std::to_string(nargs - 1)},
             {"families", family_args},
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
        if(contains(families, "read_matrix"))
            preamble += gen::generate_pointwise(*mods.at(k++), "chunk_scan_read");
        else
            preamble += stub_pointwise("chunk_scan_read");
        if(contains(families, "output_base"))
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
