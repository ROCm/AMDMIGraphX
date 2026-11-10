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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/optional.hpp>
#include <algorithm>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// out[r] = a[r] . b[indices[r]]: the rows a {rows, 1, k} times the matrix
// each row gathers from a table of matrices b {entries, k, n}, giving
// {rows, 1, n}. The inputs are the rows, the indices, the weight tensors,
// then the other inputs of the column submodule and those of the group
// submodule. fuse_ops fuses the submodules, named in `modules` in their
// order.
//
// The `weights` weight tensors share the lens of a weight space. Without the
// "weights" submodule there is one and its elements are the elements of the
// matrices; with that pointwise submodule the elements are computed from all
// of them. `layout` places the matrices in that space: it has the lens
// {entries, k, n} and its strides count elements of the weight space in
// row-major order. A weight tensor listed in `packed` stores two int4
// elements of the last axis of the weight space per byte, so it has half
// that length.
//
// The "column" pointwise submodule is applied to every sum: the sum is its
// input `column_dot` and its other `column_inputs` inputs have the lens of
// the sums. The "group" pointwise submodule then combines every `group`
// adjacent columns into one, giving rows * n / group elements: `group_args`
// gives for each of its inputs the column of the group it reads, or -1 for
// the next of its other inputs, which have the lens of its output.
//
// The other inputs of the column submodule listed in `column_tables` and
// those of the group submodule listed in `group_tables` are tables instead:
// they have the elements of one row for each of the entries, and a row reads
// those of the entry it gathers.
struct gather_dot
{
    shape layout;
    std::vector<std::size_t> packed        = {};
    std::size_t weights                    = 1;
    std::vector<std::string> modules       = {};
    std::size_t column_dot                 = 0;
    std::size_t column_inputs              = 0;
    std::vector<std::size_t> column_tables = {};
    std::size_t group                      = 1;
    std::vector<std::int64_t> group_args   = {};
    std::vector<std::size_t> group_tables  = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.layout, "layout"),
                    f(self.packed, "packed"),
                    f(self.weights, "weights"),
                    f(self.modules, "modules"),
                    f(self.column_dot, "column_dot"),
                    f(self.column_inputs, "column_inputs"),
                    f(self.column_tables, "column_tables"),
                    f(self.group, "group"),
                    f(self.group_args, "group_args"),
                    f(self.group_tables, "group_tables"));
    }

    // Whether the inputs from first listed in tables have a row of elements
    // for each of the entries
    bool has_tables(const std::vector<shape>& inputs,
                    std::size_t first,
                    const std::vector<std::size_t>& tables,
                    std::size_t row_elements) const
    {
        return std::all_of(tables.begin(), tables.end(), [&](std::size_t i) {
            return first + i < inputs.size() and
                   inputs[first + i].elements() == layout.lens().front() * row_elements;
        });
    }

    std::string name() const { return "gpu::gather_dot"; }

    // The type of the output of the last submodule applied to the sums
    shape::type_t output_type(shape::type_t sum, const std::vector<module_ref>& mods) const
    {
        if(not contains(modules, "column") and not contains(modules, "group"))
            return sum;
        return mods.back()->get_output_shapes().front().type();
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        constexpr std::size_t first = 2;
        if(mods.size() != modules.size())
            MIGRAPHX_THROW("gather_dot: expects one submodule for each of its modules");
        if(weights == 0 or inputs.size() < first + weights + column_inputs)
            MIGRAPHX_THROW("gather_dot: expects rows, indices, weights and submodule inputs");
        if(weights > 1 and not contains(modules, "weights"))
            MIGRAPHX_THROW("gather_dot: several weight tensors need the weights submodule");
        if(std::any_of(inputs.begin(), inputs.end(), [](const shape& s) { return s.dynamic(); }))
            MIGRAPHX_THROW("gather_dot: dynamic shapes are not supported");
        const auto& a       = inputs[0];
        const auto& indices = inputs[1];
        if(a.ndim() != 3 or a.lens()[1] != 1)
            MIGRAPHX_THROW("gather_dot: the rows must have the lens {rows, 1, k}");
        if(indices.ndim() != 1 or indices.lens().front() != a.lens().front())
            MIGRAPHX_THROW("gather_dot: expects one index per row");
        if(layout.ndim() != 3 or layout.lens()[1] != a.lens()[2])
            MIGRAPHX_THROW("gather_dot: the layout must have the lens {entries, k, n}");
        auto space = inputs[first].lens();
        if(contains(packed, 0))
            space.back() *= 2;
        for(std::size_t i = 0; i < weights; i++)
        {
            auto lens = inputs[first + i].lens();
            if(contains(packed, i))
                lens.back() *= 2;
            if(lens != space)
                MIGRAPHX_THROW("gather_dot: the weight tensors must share the weight space");
        }
        if(layout.element_space() > shape{layout.type(), space}.elements())
            MIGRAPHX_THROW("gather_dot: the layout reads past the weight space");
        const auto n = layout.lens()[2];
        if(group == 0 or n % group != 0)
            MIGRAPHX_THROW("gather_dot: the group must divide the columns");
        const auto first_column = first + weights;
        if(std::any_of(column_tables.begin(),
                       column_tables.end(),
                       [&](std::size_t i) { return i >= column_inputs; }) or
           not has_tables(inputs, first_column, column_tables, n) or
           not has_tables(inputs, first_column + column_inputs, group_tables, n / group))
            MIGRAPHX_THROW("gather_dot: a table must have a row of elements for each entry");
        return shape{output_type(a.type(), mods), {a.lens().front(), 1, n / group}};
    }
};
MIGRAPHX_REGISTER_OP(gather_dot);

namespace {

// The rows are listed in the local memory of a workgroup
constexpr std::size_t max_rows = 8192;

std::size_t normalize_axis(std::int64_t axis, std::size_t ndim)
{
    return axis < 0 ? axis + ndim : axis;
}

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

// Computes a row tensor {rows, ...} for the table {entries, ...} it is
// gathered from: every row must be computed from the same row of gathers
// along the first axis, which all take the same indices
struct table_lifter
{
    std::size_t rows         = 0;
    float min_rows_per_entry = 0.5f;
    // set by the first gather
    optional<instruction_ref> indices                           = nullopt;
    std::size_t entries                                         = 0;
    std::unordered_map<instruction_ref, instruction_ref> tables = {};

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
        auto axis = ins->get_operator().to_value().at("axis").to<std::int64_t>();
        if(normalize_axis(axis, dshape.ndim()) != 0)
            return nullopt;
        if(indices.has_value())
        {
            if(*indices != index or dshape.lens().front() != entries)
                return nullopt;
            return data;
        }
        // With too few rows per entry there are no rows to group
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

    // The operator computing a view of the rows for the table, if the view
    // keeps the rows apart
    optional<operation> table_view_op(instruction_ref ins) const
    {
        auto input      = ins->inputs().front();
        auto v          = ins->get_operator().to_value();
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
            if(v.at("axis").to<std::int64_t>() != 0)
                return nullopt;
            return make_op("broadcast", {{"axis", 0}, {"out_lens", table_lens(ins)}});
        }
        if(name == "transpose")
        {
            if(v.at("permutation").to_vector<std::int64_t>().front() != 0)
                return nullopt;
            return ins->get_operator();
        }
        if(contains({"squeeze", "unsqueeze", "slice"}, name))
        {
            // the axes of an unsqueeze are those of the output
            auto ndim = name == "unsqueeze" ? ins->get_shape().ndim() : input->get_shape().ndim();
            auto axes = v.at("axes").to_vector<std::int64_t>();
            if(axes.empty() or std::any_of(axes.begin(), axes.end(), [&](auto axis) {
                   return normalize_axis(axis, ndim) == 0;
               }))
                return nullopt;
            return ins->get_operator();
        }
        if(name == "unpack_int4")
        {
            if(normalize_axis(v.at("axis").to<std::int64_t>(), input->get_shape().ndim()) == 0)
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
        if(result.has_value())
            tables.emplace(ins, *result);
        return result;
    }
};

// dot(a, f(gather(t, i))) -> gather_dot(a, i, f(t)) when f computes each
// matrix from the gathered entry alone
struct find_gather_dot
{
    float min_rows_per_entry = 0.5f;

    auto matcher() const { return match::name("dot"); }

    void apply(module& m, const match::matcher_result& r) const
    {
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
        auto lens    = bs.lens();
        lens.front() = lifter.entries;
        auto dot     = m.insert_instruction(
            ins, gather_dot{shape{bs.type(), lens}}, {a, *lifter.indices, *table});
        m.replace_instruction(ins, dot);
    }
};

} // namespace

void fuse_gather_dot::apply(module_pass_manager& mpm) const
{
    if(not has_compiler_for("gpu::gather_dot"))
        return;
    match::find_matches(mpm.get_module(), find_gather_dot{min_rows_per_entry});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
