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
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>

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
MIGRAPHX_LIFT_CLASS(gather_dot_column_t, gather_dot_column);
MIGRAPHX_LIFT_CLASS(gather_dot_output_t, gather_dot_output);

struct gather_dot_settings
{
${settings}
};

extern "C" {

MIGRAPHX_GLOBAL void gather_dot_kernel(${params})
{
    make_tensors()(${args})([](auto... xs) {
        gather_dot<gather_dot_settings>(
            gather_dot_weight_t{}, gather_dot_column_t{}, gather_dot_output_t{}, xs...);
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
        return (local_memory - list_bytes) / (block * waves * a.type_size()) / 32 * 32;
    }

    // The parameters e0.. of a generated function, as declared with their
    // types or as passed
    static std::vector<std::string> extras(std::size_t n, const std::string& prefix)
    {
        std::vector<std::string> result(n);
        std::size_t i = 0;
        std::generate(result.begin(), result.end(), [&] { return prefix + std::to_string(i++); });
        return result;
    }

    // The element of the other input e of a submodule: i is the element of
    // the row and t that of the entry the row gathers, which a table holds
    static std::string
    element(const std::string& e, std::size_t pos, const std::vector<std::size_t>& tables)
    {
        return e + (contains(tables, pos) ? "[t]" : "[i]");
    }

    // name(x, i, t, e0..) returning the result
    static std::string
    function(const std::string& name, std::size_t nextras, const std::string& result)
    {
        const auto es                   = extras(nextras, "E");
        std::vector<std::string> types  = {"class X"};
        std::vector<std::string> params = {
            "X x", "[[maybe_unused]] index_int i", "[[maybe_unused]] index_int t"};
        std::transform(es.begin(), es.end(), std::back_inserter(types), [](const auto& e) {
            return "class " + e;
        });
        std::transform(es.begin(), es.end(), std::back_inserter(params), [](const auto& e) {
            return e + " " + to_lower(e);
        });
        return "template <" + join_strings(types, ", ") + ">\n__device__ auto " + name + "(" +
               join_strings(params, ", ") + ") { return " + result + "; }\n";
    }

    // name(x, i, t, e0..) calling the generated name_impl with the arguments
    static std::string
    wrapper(const std::string& name, std::size_t nextras, const std::vector<std::string>& args)
    {
        return function(name, nextras, name + "_impl(" + join_strings(args, ", ") + ")");
    }

    // gather_dot_column(x, i, t, e0..): the column submodule with the sum x
    // among the elements of its other inputs, the sum itself without the
    // submodule. The bias is the first of the inputs: the sum adds its
    // element of the entry.
    static std::string column_function(const module* m, const value& v)
    {
        const bool bias         = v.at("bias").to<bool>();
        const std::size_t first = bias ? 1 : 0;
        const std::string sum   = bias ? "(x + e0[t])" : "x";
        if(m == nullptr)
            return function("gather_dot_column", first, sum);
        const auto n      = v.at("column_inputs").to<std::size_t>();
        const auto tables = v.at("column_tables").to_vector<std::size_t>();
        const auto es     = extras(first + n, "e");
        std::vector<std::string> args;
        // the arguments so far are the other inputs before this one
        std::transform(es.begin() + first, es.end(), std::back_inserter(args), [&](const auto& e) {
            return element(e, args.size(), tables);
        });
        args.insert(args.begin() + v.at("column_dot").to<std::size_t>(), sum);
        return gen::generate_pointwise(*m, "gather_dot_column_impl") +
               wrapper("gather_dot_column", first + n, args);
    }

    // gather_dot_output(x, i, t, e0..): the group submodule with the columns
    // x[g] of a group among the elements of its other inputs, the one column
    // without the submodule
    static std::string output_function(const module* m, const value& v)
    {
        if(m == nullptr)
            return function("gather_dot_output", 0, "x[0]");
        const auto group_args = v.at("group_args").to_vector<std::int64_t>();
        const auto tables     = v.at("group_tables").to_vector<std::size_t>();
        std::size_t n         = 0;
        std::vector<std::string> args;
        std::transform(group_args.begin(),
                       group_args.end(),
                       std::back_inserter(args),
                       [&](std::int64_t column) {
                           if(column >= 0)
                               return "x[" + std::to_string(column) + "]";
                           auto arg = element("e" + std::to_string(n), n, tables);
                           n++;
                           return arg;
                       });
        return gen::generate_pointwise(*m, "gather_dot_output_impl") +
               wrapper("gather_dot_output", n, args);
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto layout  = from_value<shape>(v.at("layout"));
        const auto packed  = v.at("packed").to_vector<std::size_t>();
        const auto group   = v.at("group").to<std::size_t>();
        const auto entries = layout.lens().at(0);
        const auto k       = layout.lens().at(1);
        const auto n       = layout.lens().at(2);
        if(group == 0 or block % group != 0 or n % group != 0)
            MIGRAPHX_THROW("gpu::gather_dot: a block of columns must hold whole groups");
        // A workgroup of 16 waves with a block of columns each was the
        // fastest on gfx1201 (4, 8 and 32 were measured). Fewer are used
        // when the chunk of k would get short.
        const std::vector<std::size_t> wave_counts = {16, 8, 4, 2};
        auto fit = std::find_if(wave_counts.begin(), wave_counts.end(), [&](std::size_t w) {
            return max_chunk(w, inputs.front()) >= std::min<std::size_t>(k, 64);
        });
        const std::size_t waves = fit == wave_counts.end() ? 1 : *fit;
        hip_compile_options options;
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "gather_dot_kernel";
        // the weight tensors follow the rows and the indices
        constexpr std::size_t first_weight = 2;
        for(auto i : packed)
            options.type_overrides[first_weight + i] =
                inputs.at(first_weight + i).type() == shape::int8_type ? "migraphx::int4x2_t"
                                                                       : "migraphx::uint4x2_t";
        // the weight space with its last axis in pairs, which a packed
        // tensor stores as one element
        const std::size_t pair = packed.empty() ? 1 : 2;
        auto space_lens        = inputs.at(first_weight).lens();
        if(contains(packed, 0))
            space_lens.back() *= pair;
        space_lens.back() /= pair;
        space_lens.push_back(pair);

        // the tiles of rows that take the columns in local memory together
        constexpr std::size_t tile_batch = 4;

        std::string settings;
        settings += constant("rows", inputs.front().lens().front());
        settings += constant("entries", entries);
        settings += constant("k", k);
        settings += constant("n", n);
        settings += constant("pair", pair);
        settings += constant("weights", v.at("weights").to<std::size_t>());
        // the bias is read with the other inputs of the column function
        settings +=
            constant("column_inputs",
                     v.at("column_inputs").to<std::size_t>() + (v.at("bias").to<bool>() ? 1 : 0));
        settings += constant("group", group);
        settings += constant("col_blocks", waves);
        settings += constant("tile_batch", tile_batch);
        settings += constant("k_chunk", std::min(k, max_chunk(waves, inputs.front())));
        settings += "static constexpr auto layout = " + generate_make_shape(layout) + ";\n";
        settings += "static constexpr auto space = " +
                    generate_make_shape(shape{shape::float_type, space_lens}) + ";\n";

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

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v             = op.to_value();
        const auto& mods   = ins->module_inputs();
        const auto modules = v.at("modules").to_vector<std::string>();
        // the submodule of a role, null without it
        auto find = [&](const std::string& role) -> const module* {
            auto it = std::find(modules.begin(), modules.end(), role);
            return it == modules.end() ? nullptr : mods.at(std::distance(modules.begin(), it));
        };
        // without the weights submodule the matrices are read as they are
        const auto* weights = find("weights");
        v["preamble"] =
            (weights == nullptr
                 ? "template <class T>\n__device__ T gather_dot_weight(T x) { return x; }\n"
                 : gen::generate_pointwise(*weights, "gather_dot_weight")) +
            column_function(find("column"), v) + output_function(find("group"), v);
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
