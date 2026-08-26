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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/sym_substitute.hpp>
#include <iterator>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * The dimension the pass specializes on: the parameters that vary with it, the sizes it can take,
 * and the symbol standing for it when the module is symbolic.
 */
struct split_dimension
{
    std::vector<std::string> params;
    optional<shape::dynamic_dimension::interval> sizes;
    sym::expr symbol;
};

/**
 * Returns value if the parameters contain non-fixed dynamic_dimensions that are the same between
 * all of the dynamic shape parameters.
 * In other words, each parameter can have one non-fixed dynamic_dimension `x` where `x` is the same
 * between all of the parameters with a non-fixed dynamic_dimension.
 */
static optional<split_dimension>
find_shared_dyn_dim(const std::unordered_map<std::string, shape>& param_shapes)
{
    std::vector<std::string> params;
    std::vector<shape::dynamic_dimension> dds;
    // get non-fixed dynamic_dimension from all parameters
    for(const auto& param : param_shapes)
    {
        if(not param.second.dynamic())
            continue;
        const auto& param_dds = param.second.dyn_dims();
        auto num_non_fixed    = std::count_if(param_dds.cbegin(), param_dds.cend(), [&](auto dd) {
            if(dd.is_fixed())
                return false;
            params.push_back(param.first);
            dds.push_back(std::move(dd));
            return true;
        });
        // catch more than one non-fixed dynamic_dimension
        if(num_non_fixed > 1)
            return nullopt;
    }
    if(dds.empty())
        return nullopt;
    bool same_dd =
        std::all_of(dds.begin() + 1, dds.end(), [&](const auto& dd) { return dd == dds.front(); });
    if(not same_dd)
        return nullopt;
    // A symbolic dimension has to be specialized through its symbol, since that is what the
    // operations built from it refer to. A dimension derived from a symbol, such as `2*seq`,
    // bounds the dimension rather than the symbol, so there is nothing to substitute and it is
    // left to a caller that can name the symbol itself.
    sym::expr symbol;
    if(dds.front().is_symbolic())
    {
        const auto& e = dds.front().sym_expr;
        if(e.name() != "variable")
            return nullopt;
        symbol = sym::as_symbol(e);
    }
    // sort for a deterministic parameter order in the submodules
    std::sort(params.begin(), params.end());
    return split_dimension{std::move(params), dds.front().get_interval(), std::move(symbol)};
}

static void collect_dyn_dims(const shape& s, std::vector<shape::dynamic_dimension>& dds)
{
    const auto& subs = s.sub_shapes();
    if(not subs.empty())
    {
        std::for_each(
            subs.begin(), subs.end(), [&](const shape& sub) { collect_dyn_dims(sub, dds); });
        return;
    }
    if(not s.dynamic())
        return;
    const auto& s_dds = s.dyn_dims();
    dds.insert(dds.end(), s_dds.begin(), s_dds.end());
}

/**
 * Find the named symbol among the parameter shapes, along with every parameter that varies with
 * it. A parameter can mention the symbol in more than one dimension and can mix it with other
 * symbols; only the named one is specialized.
 *
 * The sizes the symbol can take are read off a dimension that is the symbol itself, since a
 * dimension built from it, such as `2*seq`, bounds the dimension rather than the symbol. A symbol
 * that never appears on its own leaves the sizes for the caller to name.
 *
 * Returns nothing when no parameter mentions the symbol. The pass runs over every module in the
 * program, including the specializations it just made, and only the one being split has it.
 */
static optional<split_dimension>
find_symbol_dim(const std::unordered_map<std::string, shape>& param_shapes, const std::string& name)
{
    auto symbol = sym::var(name);
    std::vector<std::string> params;
    optional<shape::dynamic_dimension::interval> sizes;
    std::for_each(param_shapes.begin(), param_shapes.end(), [&](const auto& param) {
        std::vector<shape::dynamic_dimension> dds;
        collect_dyn_dims(param.second, dds);
        auto mentions_symbol = [&](const auto& dd) {
            return dd.is_symbolic() and contains(sym::find_variables(dd.sym_expr), symbol);
        };
        if(not std::any_of(dds.begin(), dds.end(), mentions_symbol))
            return;
        params.push_back(param.first);
        auto bare = std::find_if(dds.begin(), dds.end(), [&](const auto& dd) {
            return dd.is_symbolic() and sym::same_symbol(dd.sym_expr, symbol);
        });
        if(bare != dds.end())
            sizes = bare->get_interval();
    });
    if(params.empty())
        return nullopt;
    // sort for a deterministic parameter order in the submodules
    std::sort(params.begin(), params.end());
    return split_dimension{std::move(params), sizes, std::move(symbol)};
}

/**
 * The sizes to build a submodule for, defaulting to every size the dimension can take.
 */
static std::vector<std::size_t>
specialization_sizes(std::vector<std::size_t> requested,
                     const optional<shape::dynamic_dimension::interval>& r)
{
    if(requested.empty())
    {
        if(not r.has_value())
            MIGRAPHX_THROW("split_single_dyn_dim: the sizes to specialize must be given for a "
                           "symbol with no known range");
        auto all = migraphx::range(r->min, r->max + 1);
        return {all.begin(), all.end()};
    }
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    if(r.has_value() and (requested.front() < r->min or requested.back() > r->max))
        MIGRAPHX_THROW("split_single_dyn_dim: sizes must be within [" + std::to_string(r->min) +
                       ", " + std::to_string(r->max) + "]");
    return requested;
}

/**
 * Check the parameters the split is based on to see if any of them outputs to a select_module
 * operator, which means the module has already been split.
 */
static bool any_sm_next(const_module_ref mm, const std::vector<std::string>& params)
{
    return std::any_of(params.begin(), params.end(), [&](const auto& param) {
        auto outputs = mm->get_parameter(param)->outputs();
        return std::any_of(outputs.cbegin(), outputs.cend(), [](auto ins) {
            return ins->name() == "select_module";
        });
    });
}

/**
 * Maps every literal in `mm` to itself. Seeding a submodule's instruction map with this leaves
 * the literals in the parent module for the submodule to capture, instead of copying one set
 * per specialization.
 */
static std::unordered_map<instruction_ref, instruction_ref> literal_captures(const_module_ref mm)
{
    std::unordered_map<instruction_ref, instruction_ref> captures;
    auto instructions = iterator_for(*mm);
    transform_if(
        instructions.begin(),
        instructions.end(),
        std::inserter(captures, captures.end()),
        [](instruction_ref ins) { return ins->name() == "@literal"; },
        [](instruction_ref ins) { return std::make_pair(ins, ins); });
    return captures;
}

/**
 * Build one submodule per size the split dimension takes, each specialized to that size, and
 * replace the module's body with a select_module that picks between them. Probably won't work for
 * `if` and `loop` instructions, depending on how the submodules for those work.
 */
static std::vector<module_ref> make_specializations(module_pass_manager& mpm,
                                                    module_ref mm,
                                                    const split_dimension& dim,
                                                    const std::vector<std::size_t>& sizes)
{
    const auto captures = literal_captures(mm);
    std::vector<module_ref> submodules;
    std::transform(
        sizes.begin(), sizes.end(), std::back_inserter(submodules), [&](std::size_t size) {
            auto* submod = mpm.create_module("dim_" + std::to_string(size));
            // instruction map for new static shaped submodule parameters
            std::unordered_map<instruction_ref, instruction_ref> map_ins = captures;
            symbol_map symbols;
            if(not dim.symbol.empty())
                symbols.emplace(dim.symbol, size);
            std::transform(dim.params.begin(),
                           dim.params.end(),
                           std::inserter(map_ins, map_ins.end()),
                           [&](const auto& name) {
                               auto s = mm->get_parameter_shape(name);
                               // Substituting the symbol leaves any other symbol in the shape
                               // alone, so a parameter can stay dynamic in another dimension.
                               auto static_shape = symbols.empty() ? s.to_static(size)
                                                                   : substitute_symbols(s, symbols);
                               return std::make_pair(mm->get_parameter(name),
                                                     submod->add_parameter(name, static_shape));
                           });
            // Static inputs make most shapes fall out on their own, but an operation holding a
            // symbol in an attribute has to be told what the symbol is worth here.
            module::inserter insert = nullptr;
            if(not symbols.empty())
                insert = [symbols](module& m,
                                   instruction_ref pos,
                                   const operation& op,
                                   const std::vector<instruction_ref>& inputs,
                                   const std::vector<module_ref>& mod_args) {
                    return m.insert_instruction(pos, op.to_static(symbols), inputs, mod_args);
                };
            submod->add_return(submod->add_instructions(mm, &map_ins, std::move(insert)));
            return submod;
        });
    return submodules;
}

void split_single_dyn_dim::apply(module_pass_manager& mpm) const
{
    module_ref mm     = &mpm.get_module();
    auto param_shapes = mm->get_parameter_shapes();
    auto dim =
        symbol.empty() ? find_shared_dyn_dim(param_shapes) : find_symbol_dim(param_shapes, symbol);
    if(not dim.has_value() or any_sm_next(mm, dim->params))
        return;

    auto submodules = make_specializations(mpm, mm, *dim, specialization_sizes(sizes, dim->sizes));

    // sort parameters by name for consistency (vs. parameter order attr)
    auto param_names = mm->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    // redirect to select_module operator and return
    std::vector<instruction_ref> sm_inputs;
    std::transform(param_names.cbegin(),
                   param_names.cend(),
                   std::back_inserter(sm_inputs),
                   [&](auto pn) { return mm->get_parameter(std::move(pn)); });
    // The main module keeps its original shapes, so these are the exact output shapes for
    // whatever size shows up at runtime, symbols and all.
    auto output_shapes = mm->get_output_shapes();
    auto sm_ins        = mm->add_instruction(
        migraphx::make_op("select_module",
                                 {{"output_dyn_shapes", migraphx::to_value(shape{output_shapes})}}),
        sm_inputs,
        submodules);
    auto indices = migraphx::range(output_shapes.size());
    std::vector<instruction_ref> outputs;
    std::transform(indices.begin(), indices.end(), std::back_inserter(outputs), [&](std::size_t i) {
        return mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", i}}), sm_ins);
    });
    mm->replace_return(outputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
