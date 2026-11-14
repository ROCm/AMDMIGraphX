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
 *
 */

#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/fuse_gather_dot.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const gather_dot_kernel = R"__migraphx__(
#include <migraphx/kernels/gather_dot.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

MIGRAPHX_LIFT_CLASS(gather_dot_weight_t, gather_dot_weight);
MIGRAPHX_LIFT_CLASS(gather_dot_output_t, gather_dot_output);

struct gather_dot_settings
{
${settings}
};

extern "C" {

MIGRAPHX_GLOBAL void gather_dot_kernel(${params})
{
    make_tensors()(${args})([](auto... xs) {
        gather_dot<gather_dot_settings>(gather_dot_weight_t{}, gather_dot_output_t{}, xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct gather_dot_compiler : compiler<gather_dot_compiler>
{
    std::vector<std::string> names() const { return {"gpu::gather_dot"}; }

    static std::string constant(const std::string& name, std::size_t x)
    {
        return "static constexpr index_int " + name + " = " + std::to_string(x) + ";\n";
    }

    // the rows and the columns of one matrix instruction
    static constexpr std::size_t block = 16;

    // The longest chunk of k whose columns fit in local memory beside the
    // list of rows when a workgroup has a block of columns for each of its
    // waves, in whole runs of 32
    static std::size_t max_chunk(std::size_t waves, const shape& a)
    {
        constexpr std::size_t local_memory = 64 * 1024;
        const auto list_bytes              = (a.lens().front() + 3 * block) * sizeof(std::uint32_t);
        // fuse_gather_dot limits the rows so their list leaves room
        assert(list_bytes < local_memory);
        return (local_memory - list_bytes) / (block * waves * a.type_size()) / 32 * 32;
    }

    // gather_dot_output(x, i, e0..) returning the result: x holds the columns
    // of a group and i is the element of the nextras other inputs e0..
    static std::string output_returning(std::size_t nextras, const std::string& result)
    {
        std::vector<std::string> types  = {"class X"};
        std::vector<std::string> params = {"X x", "[[maybe_unused]] index_int i"};
        const auto es                   = range(nextras);
        std::transform(es.begin(), es.end(), std::back_inserter(types), [](auto e) {
            return "class E" + std::to_string(e);
        });
        std::transform(es.begin(), es.end(), std::back_inserter(params), [](auto e) {
            return "E" + std::to_string(e) + " e" + std::to_string(e);
        });
        return "template <" + join_strings(types, ", ") + ">\n__device__ auto gather_dot_output(" +
               join_strings(params, ", ") + ") { return " + result + "; }\n";
    }

    // The output function: each arg of the pointwise operator reads a column
    // of the group or the next other input. Without a submodule it returns
    // the one column.
    static std::string output_function(const_module_ref m,
                                       const std::vector<optional<std::size_t>>& args)
    {
        if(m == nullptr)
            return output_returning(0, "x[0]");
        std::size_t n = 0;
        std::vector<std::string> call_args;
        std::transform(args.begin(),
                       args.end(),
                       std::back_inserter(call_args),
                       [&](const optional<std::size_t>& column) {
                           if(column.has_value())
                               return "x[" + std::to_string(*column) + "]";
                           return "e" + std::to_string(n++) + "[i]";
                       });
        return gen::generate_pointwise(*m, "gather_dot_output_impl") +
               output_returning(n, "gather_dot_output_impl(" + join_strings(call_args, ", ") + ")");
    }

    static std::size_t product(const std::vector<std::size_t>& lens, std::size_t first)
    {
        return std::accumulate(
            lens.begin() + first, lens.end(), std::size_t{1}, std::multiplies<>{});
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto matrices = v.at("matrices").to_vector<std::size_t>();
        const auto packed   = v.at("packed").to_vector<std::size_t>();
        const auto group    = v.at("group").to<std::size_t>();
        const auto nweights = v.at("weights").to<std::size_t>();
        const auto& a       = inputs.front();
        // the matrices are {entries, k axes..., n axes...} for rows {rows, k axes...}
        const auto run_axis = a.ndim() - 1;
        const auto entries  = matrices.front();
        const auto k        = product(a.lens(), 1);
        const auto n        = product(matrices, a.ndim());
        if(group == 0 or block % group != 0 or n % group != 0)
            MIGRAPHX_THROW("gpu::gather_dot: a block of columns must hold whole groups");
        // A workgroup of 16 waves with a block of columns each was the
        // fastest on gfx1201 (4, 8 and 32 were measured). Fewer are used
        // when the chunk of k would get short.
        const std::vector<std::size_t> wave_counts = {16, 8, 4, 2};
        auto fit = std::find_if(wave_counts.begin(), wave_counts.end(), [&](std::size_t w) {
            return max_chunk(w, a) >= std::min<std::size_t>(k, 64);
        });
        const std::size_t waves = fit == wave_counts.end() ? 1 : *fit;
        const auto chunk        = max_chunk(waves, a);
        hip_compile_options options;
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "gather_dot_kernel";
        constexpr auto first_weight = gather_dot_layout::first_weight;
        std::transform(packed.begin(),
                       packed.end(),
                       std::inserter(options.type_overrides, options.type_overrides.end()),
                       [&](std::size_t i) {
                           const bool is_signed =
                               inputs.at(first_weight + i).type() == shape::int8_type;
                           return std::make_pair(first_weight + i,
                                                 std::string{is_signed ? "migraphx::int4x2_t"
                                                                       : "migraphx::uint4x2_t"});
                       });
        // the lens of the matrices with the axis the bytes are unpacked
        // along in pairs, which a packed tensor stores as one element
        const std::size_t pair = packed.empty() ? 1 : 2;
        const std::size_t pair_axis =
            packed.empty() ? run_axis : v.at("unpack_axis").to<std::size_t>();
        if(pair_axis == 0)
            MIGRAPHX_THROW("gpu::gather_dot: the entries cannot be unpacked");
        auto space_lens = matrices;
        space_lens.at(pair_axis) /= pair;
        space_lens.insert(space_lens.begin() + pair_axis + 1, pair);
        // the k axes of the space, with the pairs when they are along one
        const auto k_axes = run_axis + (pair_axis <= run_axis ? 1 : 0);
        const std::vector<std::size_t> k_lens(space_lens.begin() + 1,
                                              space_lens.begin() + 1 + k_axes);
        const std::vector<std::size_t> n_lens(space_lens.begin() + 1 + k_axes, space_lens.end());
        // The weight tensors are loaded a run of k at a time when they store
        // the innermost k axis adjacent or broadcast, and the bytes of a
        // packed tensor are whole pairs of that axis
        const bool run_loads =
            pair_axis == run_axis and
            std::all_of(inputs.begin() + first_weight,
                        inputs.begin() + first_weight + nweights,
                        [&](const shape& s) { return s.strides().at(run_axis) <= 1; });

        std::string settings;
        settings += constant("rows", a.lens().front());
        settings += constant("entries", entries);
        settings += constant("k", k);
        settings += constant("n", n);
        settings += constant("pair", pair);
        settings += constant("pair_axis", pair_axis);
        settings += constant("run_axis", run_axis);
        settings += constant("run_len", matrices.at(run_axis));
        settings += constant("run_loads", run_loads ? 1 : 0);
        settings += constant("weights", nweights);
        settings += constant("bias", v.at("bias").to<bool>() ? 1 : 0);
        settings += constant("group", group);
        settings += constant("col_blocks", waves);
        settings += constant("k_chunk", std::min(k, chunk));
        settings += "static constexpr auto k_space = " +
                    generate_make_shape(shape{shape::float_type, k_lens}) + ";\n";
        settings += "static constexpr auto n_space = " +
                    generate_make_shape(shape{shape::float_type, n_lens}) + ";\n";

        // one workgroup for each entry and each of its sets of columns
        const auto nlocal  = waves * ctx.get_current_device().get_wavefront_size();
        const auto columns = block * waves;
        const auto groups  = entries * ((n + columns - 1) / columns);
        options.set_launch_params(v, groups * nlocal, nlocal);
        auto src = interpolate_string(gather_dot_kernel,
                                      {{"preamble", v.at("preamble").to<std::string>()},
                                       {"settings", settings},
                                       {"params", enum_params(inputs.size(), "void * p")},
                                       {"args", enum_params(inputs.size(), "p")}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation&) const
    {
        // the weights submodule, then the output submodule if there is one
        const auto& mods   = ins->module_inputs();
        const auto inputs  = to_shapes(ins->inputs());
        const auto weights = read_gather_dot_weights(*mods.front());
        const auto output =
            mods.size() == 2 ? read_gather_dot_output(*mods.back()) : gather_dot_output{};
        // the output buffer follows the inputs
        const auto layout = get_gather_dot_layout(inputs.size() - 1, mods);
        assert(layout.size() + 1 == inputs.size());
        value v          = value::object{};
        v["weights"]     = layout.weights;
        v["bias"]        = layout.bias;
        v["matrices"]    = to_value(mods.front()->get_output_shapes().front().lens());
        v["packed"]      = to_value(weights.packed);
        v["unpack_axis"] = weights.unpack_axis;
        v["group"]       = output.group;
        v["preamble"] =
            (weights.function == nullptr
                 ? "template <class T>\n__device__ T gather_dot_weight(T x) { return x; }\n"
                 : gen::generate_pointwise(*weights.function, "gather_dot_weight")) +
            output_function(output.function, output.args);
        return compile_op(ctx, inputs, v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
