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
#include <migraphx/tune_axis.hpp>
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
    static std::string output_function(const module* m,
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

    // What the kernel needs from the weights submodule
    struct weights_plan
    {
        // the weight tensors that store two int4 elements per byte
        std::vector<std::size_t> packed = {};
        // the axis they are unpacked along
        std::size_t unpack_axis = 0;
        // the pointwise submodule computing an element of the matrices
        const module* function = nullptr;
    };

    // The instruction a submodule returns
    static instruction_ref result_of(const module& m)
    {
        const auto rets = m.get_returns();
        if(rets.size() != 1)
            MIGRAPHX_THROW("gpu::gather_dot: a submodule must return one result");
        return rets.front();
    }

    // The weights submodule returns its parameter, which is the table, or
    // a pointwise operator of its parameters in their order, of which some
    // are read through an unpack_int4
    static weights_plan read_weights(const module& m)
    {
        const auto names = m.get_parameter_names();
        auto pw          = result_of(m);
        if(pw->name() == "@param" and names.size() == 1)
            return {};
        if(pw->name() != "pointwise" or pw->inputs().size() != names.size())
            MIGRAPHX_THROW("gpu::gather_dot: the weights submodule must return a pointwise "
                           "operator of its parameters");
        // the axis each parameter is unpacked along
        std::vector<optional<std::size_t>> axes;
        std::transform(names.begin(),
                       names.end(),
                       pw->inputs().begin(),
                       std::back_inserter(axes),
                       [&](const std::string& name, instruction_ref arg) -> optional<std::size_t> {
                           auto param = m.get_parameter(name);
                           if(arg == param)
                               return nullopt;
                           if(arg->name() != "unpack_int4" or arg->inputs().front() != param)
                               MIGRAPHX_THROW("gpu::gather_dot: the weights submodule must read "
                                              "its parameters in order");
                           auto axis = arg->get_operator().to_value().at("axis").to<int>();
                           return tune_axis(param->get_shape().ndim(), axis, arg->name());
                       });
        weights_plan result;
        result.function = pw->module_inputs().front();
        const auto is   = range(axes.size());
        std::copy_if(is.begin(), is.end(), std::back_inserter(result.packed), [&](std::size_t i) {
            return axes[i].has_value();
        });
        if(result.packed.empty())
            return result;
        result.unpack_axis = *axes[result.packed.front()];
        if(std::any_of(result.packed.begin(), result.packed.end(), [&](std::size_t i) {
               return *axes[i] != result.unpack_axis;
           }))
            MIGRAPHX_THROW("gpu::gather_dot: the weight tensors must be unpacked along one axis");
        return result;
    }

    // What the kernel needs from the output submodule
    struct output_plan
    {
        // the columns of a group
        std::size_t group = 1;
        // the column of the group each input of the pointwise operator
        // reads, nothing for the next of the other inputs
        std::vector<optional<std::size_t>> args = {};
        // the number of other inputs
        std::size_t others = 0;
        // the pointwise submodule computing an element of the result
        const module* function = nullptr;
    };

    // The output submodule returns a pointwise operator of columns of the
    // sums, its first parameter, and of its other parameters in their order
    static output_plan read_output(const module& m)
    {
        const auto names = m.get_parameter_names();
        auto pw          = result_of(m);
        if(pw->name() != "pointwise")
            MIGRAPHX_THROW("gpu::gather_dot: the output submodule must return a pointwise "
                           "operator of its parameters");
        auto sums = m.get_parameter(names.front());
        output_plan result;
        result.function = pw->module_inputs().front();
        std::vector<std::size_t> groups;
        std::vector<instruction_ref> others;
        std::transform(pw->inputs().begin(),
                       pw->inputs().end(),
                       std::back_inserter(result.args),
                       [&](instruction_ref input) -> optional<std::size_t> {
                           auto column = find_gather_dot_column(input);
                           if(not column.has_value() or column->sums != sums)
                           {
                               others.push_back(input);
                               return nullopt;
                           }
                           groups.push_back(column->group);
                           return column->column;
                       });
        if(groups.empty() or
           std::adjacent_find(groups.begin(), groups.end(), std::not_equal_to<>{}) != groups.end())
            MIGRAPHX_THROW("gpu::gather_dot: the output submodule must read columns of the sums "
                           "in groups of one size");
        if(others.size() + 1 != names.size() or
           not std::equal(others.begin(),
                          others.end(),
                          names.begin() + 1,
                          [&](instruction_ref input, const std::string& name) {
                              return input == m.get_parameter(name);
                          }))
            MIGRAPHX_THROW("gpu::gather_dot: the output submodule must read its other "
                           "parameters in order");
        result.group  = groups.front();
        result.others = others.size();
        return result;
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
        // the weight tensors follow the rows and the indices
        constexpr std::size_t first_weight = 2;
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
        settings += constant("bias", v.at("bias").to<std::size_t>());
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
        const auto& mods    = ins->module_inputs();
        const auto inputs   = to_shapes(ins->inputs());
        const auto weights  = read_weights(*mods.front());
        const auto output   = mods.size() == 2 ? read_output(*mods.back()) : output_plan{};
        const auto nweights = mods.front()->get_parameter_names().size();
        // the rows, the indices and the output buffer
        constexpr std::size_t fixed = 3;
        // the input no submodule accounts for
        const std::size_t bias = inputs.size() - fixed - nweights - output.others;
        assert(bias <= 1);
        value v          = value::object{};
        v["weights"]     = nweights;
        v["bias"]        = bias;
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
