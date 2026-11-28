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
#include <migraphx/gpu/fuse_gather_dot.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/tune_axis.hpp>
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// out[r, n...] = sum over k... of a[r, k...] * b[indices[r], k..., n...] for
// rows a {rows, k axes...} and a table of matrices b
// {entries, k axes..., n axes...}. The inputs are the rows, the indices, the
// weight tensors, the bias if one input is left over, then the other inputs
// of the output submodule.
//
// The weights submodule computes the table from the weight tensors: it
// returns its parameter, or a pointwise operator of its parameters, some
// through an unpack_int4. The bias holds n elements per entry. The output
// submodule, if there is one, is a pointwise operator of columns of the sums,
// its first parameter, and of its other parameters.
struct gather_dot
{
    std::string name() const { return "gpu::gather_dot"; }

    // Whether the shapes from first have the lens of the named parameters
    // of the submodule
    static bool matches_parameters(const module& m,
                                   std::vector<std::string>::const_iterator names_first,
                                   std::vector<std::string>::const_iterator names_last,
                                   std::vector<shape>::const_iterator first)
    {
        const auto shapes = m.get_parameter_shapes();
        return std::equal(
            names_first, names_last, first, [&](const std::string& name, const shape& s) {
                return shapes.at(name).lens() == s.lens();
            });
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        if(mods.empty() or mods.size() > 2)
            MIGRAPHX_THROW("gather_dot: expects the weights submodule, then the output submodule "
                           "if there is one");
        const auto names  = mods.front()->get_parameter_names();
        const auto onames = mods.back()->get_parameter_names();
        if(names.empty() or onames.empty())
            MIGRAPHX_THROW("gather_dot: a submodule has no parameters");
        const auto layout = get_gather_dot_layout(inputs.size(), mods);
        if(inputs.size() != layout.size())
            MIGRAPHX_THROW("gather_dot: expects rows, indices, weights, bias and the other inputs "
                           "of the output submodule");
        if(std::any_of(inputs.begin(), inputs.end(), [](const shape& s) { return s.dynamic(); }))
            MIGRAPHX_THROW("gather_dot: dynamic shapes are not supported");
        const auto& a       = inputs[gather_dot_layout::rows];
        const auto& indices = inputs[gather_dot_layout::indices];
        if(a.ndim() < 2)
            MIGRAPHX_THROW("gather_dot: the rows must have the lens {rows, k axes...}");
        if(indices.ndim() != 1 or indices.lens().front() != a.lens().front())
            MIGRAPHX_THROW("gather_dot: expects one index per row");
        if(not matches_parameters(*mods.front(),
                                  names.begin(),
                                  names.end(),
                                  inputs.begin() + gather_dot_layout::first_weight))
            MIGRAPHX_THROW(
                "gather_dot: the weight tensors must have the lens of the parameters of the "
                "weights submodule");
        // the kernel is generated from the submodules, so they must be ones
        // it can read
        read_gather_dot_weights(*mods.front());
        const auto tables = mods.front()->get_output_shapes();
        if(tables.size() != 1)
            MIGRAPHX_THROW("gather_dot: the weights submodule must return the table");
        const auto& matrices = tables.front().lens();
        if(matrices.size() < a.ndim() + 1 or
           not std::equal(a.lens().begin() + 1, a.lens().end(), matrices.begin() + 1))
            MIGRAPHX_THROW(
                "gather_dot: the matrices must have the lens {entries, k axes..., n axes...}");
        std::vector<std::size_t> lens = {a.lens().front()};
        lens.insert(lens.end(), matrices.begin() + a.ndim(), matrices.end());
        const auto n = std::accumulate(
            matrices.begin() + a.ndim(), matrices.end(), std::size_t{1}, std::multiplies<>{});
        if(layout.bias and (inputs[layout.bias_index()].type() != a.type() or
                            inputs[layout.bias_index()].elements() != matrices.front() * n))
            MIGRAPHX_THROW(
                "gather_dot: the bias must have the type of the rows and n elements per entry");
        if(mods.size() == 1)
            return shape{a.type(), lens};
        const auto& output = *mods.back();
        if(output.get_parameter_shapes().at(onames.front()).lens() != lens or
           not matches_parameters(
               output, onames.begin() + 1, onames.end(), inputs.begin() + layout.first_other()))
            MIGRAPHX_THROW("gather_dot: the output submodule must take the sums, then inputs "
                           "with the lens of its other parameters");
        read_gather_dot_output(output);
        const auto results = output.get_output_shapes();
        if(results.size() != 1)
            MIGRAPHX_THROW("gather_dot: the output submodule must return one result");
        return results.front().as_standard();
    }
};
MIGRAPHX_REGISTER_OP(gather_dot);

// The tensor a view reads in its order, through views nothing else reads
static optional<instruction_ref> find_viewed(instruction_ref view)
{
    // the views of a tensor that keep it standard keep its order
    if(not view->get_shape().standard())
        return nullopt;
    auto path = get_input_path(view);
    auto it   = std::find_if_not(path.begin(), path.end(), [](instruction_ref ins) {
        return contains({"reshape_lazy", "squeeze", "unsqueeze"}, ins->name()) and
               ins->inputs().front()->outputs().size() == 1;
    });
    if(it == path.end())
        return nullopt;
    return *it;
}

optional<gather_dot_column> find_gather_dot_column(instruction_ref ins)
{
    auto path = get_input_path(ins);
    auto it   = std::find_if_not(path.begin(), path.end(), [](instruction_ref x) {
        return contains({"squeeze", "unsqueeze"}, x->name()) and x->outputs().size() == 1;
    });
    if(it == path.end() or (*it)->name() != "slice" or (*it)->inputs().size() != 1 or
       (*it)->outputs().size() != 1)
    {
        auto sums = find_viewed(ins);
        if(not sums.has_value())
            return nullopt;
        return gather_dot_column{*sums, ins};
    }
    auto slice     = *it;
    auto v         = slice->get_operator().to_value();
    auto axes      = v.at("axes").to_vector<std::size_t>();
    auto starts    = v.at("starts").to_vector<std::size_t>();
    auto ends      = v.at("ends").to_vector<std::size_t>();
    auto view      = slice->inputs().front();
    const auto& vs = view->get_shape();
    if(axes.size() != 1 or axes.front() + 1 != vs.ndim() or ends.front() != starts.front() + 1)
        return nullopt;
    auto sums = find_viewed(view);
    if(not sums.has_value())
        return nullopt;
    return gather_dot_column{*sums, view, slice, starts.front(), vs.lens().back()};
}

gather_dot_layout get_gather_dot_layout(std::size_t ninputs, const std::vector<module_ref>& mods)
{
    assert(not mods.empty());
    gather_dot_layout result;
    result.weights = mods.front()->get_parameter_names().size();
    // the output submodule takes the sums, then its other inputs
    if(mods.size() > 1)
    {
        const auto nparams = mods.back()->get_parameter_names().size();
        assert(nparams > 0);
        result.others = nparams - 1;
    }
    // the input no submodule accounts for
    result.bias = ninputs > result.size();
    return result;
}

// The instruction a submodule returns
static instruction_ref result_of(const module& m)
{
    const auto rets = m.get_returns();
    if(rets.size() != 1)
        MIGRAPHX_THROW("gpu::gather_dot: a submodule must return one result");
    return rets.front();
}

gather_dot_weights read_gather_dot_weights(const module& m)
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
    gather_dot_weights result;
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

gather_dot_output read_gather_dot_output(const module& m)
{
    const auto names = m.get_parameter_names();
    auto pw          = result_of(m);
    if(pw->name() != "pointwise")
        MIGRAPHX_THROW("gpu::gather_dot: the output submodule must return a pointwise "
                       "operator of its parameters");
    auto sums = m.get_parameter(names.front());
    gather_dot_output result;
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
    result.group = groups.front();
    return result;
}

namespace {

// The rows are listed in the local memory of a workgroup
constexpr std::size_t max_rows = 8192;

bool is_pointwise(instruction_ref ins)
{
    return ins->name() == "pointwise" or ins->get_operator().attributes().contains("pointwise");
}

// Whether the input is one row broadcast to all the rows
bool is_row_broadcast(instruction_ref input)
{
    if(input->name() != "multibroadcast")
        return false;
    const auto& s = input->inputs().front()->get_shape();
    if(s.dynamic())
        return false;
    return s.ndim() < input->get_shape().ndim() or s.lens().front() == 1;
}

// Rebuilds a tensor {rows, ...} computed from gathers along the first axis
// as the table {entries, ...} it would be gathered from. Each row must depend
// only on its own gathered entry, and all the gathers take the same indices.
struct table_lifter
{
    std::size_t rows         = 0;
    float min_rows_per_entry = 0.0f;
    // set by the first gather
    optional<instruction_ref> indices = nullopt;
    std::size_t entries               = 0;
    // what each instruction lifts to, nothing when it cannot be lifted
    std::unordered_map<instruction_ref, optional<instruction_ref>> tables = {};

    std::vector<std::size_t> table_lens(instruction_ref ins) const
    {
        auto lens    = ins->get_shape().lens();
        lens.front() = entries;
        return lens;
    }

    optional<instruction_ref> lift_gather(instruction_ref ins)
    {
        auto data          = ins->inputs().front();
        auto index         = ins->inputs().back();
        const auto& dshape = data->get_shape();
        const auto& ishape = index->get_shape();
        if(dshape.dynamic() or ishape.dynamic() or ishape.ndim() != 1)
            return nullopt;
        auto axis = ins->get_operator().to_value().at("axis").to<int>();
        if(tune_axis(dshape.ndim(), axis, ins->name()) != 0)
            return nullopt;
        if(indices.has_value())
        {
            if(*indices != index or dshape.lens().front() != entries)
                return nullopt;
            return data;
        }
        // With one entry or too few rows per entry, grouping by entry saves nothing
        if(dshape.lens().front() < 2 or rows < min_rows_per_entry * dshape.lens().front())
            return nullopt;
        indices = index;
        entries = dshape.lens().front();
        return data;
    }

    optional<instruction_ref> lift_pointwise(module& m, instruction_ref pos, instruction_ref ins)
    {
        const auto& lens   = ins->get_shape().lens();
        const auto& inputs = ins->inputs();
        if(not std::all_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
               return not input->get_shape().dynamic() and input->get_shape().lens() == lens;
           }))
            return nullopt;
        std::vector<optional<instruction_ref>> lifted;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(lifted),
                       [&](instruction_ref input) { return lift(m, pos, input); });
        if(std::none_of(
               lifted.begin(), lifted.end(), [](const auto& table) { return table.has_value(); }))
            return nullopt;
        // The other inputs must be one row broadcast to all the rows
        if(not std::equal(inputs.begin(),
                          inputs.end(),
                          lifted.begin(),
                          [](instruction_ref input, const auto& table) {
                              return table.has_value() or is_row_broadcast(input);
                          }))
            return nullopt;
        auto bcast = make_op("multibroadcast", {{"out_lens", table_lens(ins)}});
        std::vector<instruction_ref> args;
        std::transform(inputs.begin(),
                       inputs.end(),
                       lifted.begin(),
                       std::back_inserter(args),
                       [&](instruction_ref input, const auto& table) {
                           if(table.has_value())
                               return *table;
                           return m.insert_instruction(pos, bcast, input->inputs().front());
                       });
        return m.insert_instruction(pos, ins->get_operator(), args, ins->module_inputs());
    }

    // The operator that applies the same view to the table, if the view
    // leaves the first axis alone
    optional<operation> table_view_op(instruction_ref ins) const
    {
        auto input      = ins->inputs().front();
        const auto name = ins->name();
        if(name == "contiguous")
            return ins->get_operator();
        if(name == "reshape")
            return make_op("reshape", {{"dims", table_lens(ins)}});
        if(name == "multibroadcast")
        {
            if(input->get_shape().ndim() != ins->get_shape().ndim())
                return nullopt;
            return make_op("multibroadcast", {{"out_lens", table_lens(ins)}});
        }
        if(name == "broadcast")
        {
            if(ins->get_operator().to_value().at("axis").to<std::int64_t>() != 0)
                return nullopt;
            return make_op("broadcast", {{"axis", 0}, {"out_lens", table_lens(ins)}});
        }
        if(name == "transpose")
        {
            auto v = ins->get_operator().to_value();
            if(v.at("permutation").to_vector<std::int64_t>().front() != 0)
                return nullopt;
            return ins->get_operator();
        }
        if(contains({"squeeze", "unsqueeze", "slice"}, name))
        {
            // the axes of an unsqueeze are those of the output
            auto ndim = name == "unsqueeze" ? ins->get_shape().ndim() : input->get_shape().ndim();
            auto axes = ins->get_operator().to_value().at("axes").to_vector<int>();
            if(axes.empty() or std::any_of(axes.begin(), axes.end(), [&](int axis) {
                   return tune_axis(ndim, axis, name) == 0;
               }))
                return nullopt;
            return ins->get_operator();
        }
        if(name == "unpack_int4")
        {
            auto axis = ins->get_operator().to_value().at("axis").to<int>();
            if(tune_axis(input->get_shape().ndim(), axis, name) == 0)
                return nullopt;
            return ins->get_operator();
        }
        return nullopt;
    }

    optional<instruction_ref> lift_view(module& m, instruction_ref pos, instruction_ref ins)
    {
        auto table = lift(m, pos, ins->inputs().front());
        if(not table.has_value())
            return nullopt;
        auto op = table_view_op(ins);
        if(not op.has_value())
            return nullopt;
        return m.insert_instruction(pos, *op, *table);
    }

    optional<instruction_ref> lift(module& m, instruction_ref pos, instruction_ref ins)
    {
        auto it = tables.find(ins);
        if(it != tables.end())
            return it->second;
        const auto& s = ins->get_shape();
        if(s.dynamic() or s.ndim() == 0 or s.lens().front() != rows)
            return nullopt;
        optional<instruction_ref> result = nullopt;
        if(ins->name() == "gather")
            result = lift_gather(ins);
        else if(is_pointwise(ins))
            result = lift_pointwise(m, pos, ins);
        else if(ins->inputs().size() == 1)
            result = lift_view(m, pos, ins);
        tables.emplace(ins, result);
        return result;
    }
};

// dot(a, f(gather(t, i))) -> gather_dot(a, i, f(t)) when f computes each
// matrix from the gathered entry alone
struct find_gather_dot
{
    float min_rows_per_entry = 0.0f;
    // the number of weights submodules added, to name them
    std::size_t* counter = nullptr;

    auto matcher() const { return match::name("dot"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m        = mpm.get_module();
        auto ins       = r.result;
        auto a         = ins->inputs().front();
        auto b         = ins->inputs().back();
        const auto& as = a->get_shape();
        const auto& bs = b->get_shape();
        if(as.dynamic() or bs.dynamic() or as.ndim() != 3 or as.lens()[1] != 1)
            return;
        // the kernel accumulates the products in floating point
        if(not contains({shape::half_type, shape::bf16_type, shape::float_type, shape::double_type},
                        as.type()))
            return;
        const auto rows = as.lens().front();
        if(rows > max_rows)
            return;
        table_lifter lifter{rows, min_rows_per_entry};
        auto table = lifter.lift(m, ins, b);
        if(not table.has_value())
            return;
        // a table is lifted from at least one gather
        assert(lifter.indices.has_value());
        // the weights submodule returns the table
        auto* wm = mpm.create_module(m.name() + ":gather_dot" + std::to_string((*counter)++));
        wm->set_bypass();
        wm->add_return({wm->add_parameter(param_name(0), (*table)->get_shape().as_standard())});
        // the rows {rows, k} times the matrices {entries, k, n} give {rows, n}
        auto vectors = m.insert_instruction(ins, make_op("squeeze", {{"axes", {1}}}), a);
        auto dot =
            m.insert_instruction(ins, gather_dot{}, {vectors, *lifter.indices, *table}, {wm});
        m.replace_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), dot);
    }
};

// add(gather_dot(a, i, t), f(gather(b, i))) -> gather_dot(a, i, t, f(b)) when
// f computes the bias of each row from the gathered entry alone. The add is
// folded here since the pointwise fusion would merge it with what reads it.
struct find_gather_dot_bias
{
    auto matcher() const
    {
        // the sums are read through the unsqueeze find_gather_dot left
        auto sums = match::name("unsqueeze")(
            match::used_once(),
            match::arg(0)(
                match::name("gpu::gather_dot")(match::used_once(), match::nargs(3)).bind("dot")));
        return match::name("add")(
            match::either_arg(0, 1)(sums.bind("sums"), match::any().bind("bias")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m            = mpm.get_module();
        auto ins           = r.result;
        auto dot           = r.instructions["dot"];
        auto sums          = r.instructions["sums"];
        auto bias          = r.instructions["bias"];
        const auto& s      = sums->get_shape();
        const auto& inputs = dot->inputs();
        const auto& bs     = bias->get_shape();
        // only the sums as find_gather_dot left them, without a bias
        if(dot->module_inputs().size() != 1)
            return;
        if(bs.dynamic() or bs.lens() != s.lens() or bs.type() != s.type())
            return;
        const auto layout = get_gather_dot_layout(inputs.size(), dot->module_inputs());
        auto entries      = inputs.at(gather_dot_layout::first_weight)->get_shape().lens().front();
        // any number of rows per entry, as the rows are grouped already
        table_lifter lifter{s.lens().front(), 0.0f, inputs.at(gather_dot_layout::indices), entries};
        auto table = lifter.lift(m, ins, bias);
        if(not table.has_value())
            return;
        // the bias follows the table
        auto args = inputs;
        args.insert(args.begin() + layout.bias_index(), *table);
        auto biased = m.insert_instruction(ins, dot->get_operator(), args, dot->module_inputs());
        m.replace_instruction(ins, sums->get_operator(), biased);
    }
};

} // namespace

void fuse_gather_dot::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    // In the order of the instructions: the dot, then the add of its bias
    match::find_matches(mpm, find_gather_dot{min_rows_per_entry, &counter}, find_gather_dot_bias{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
