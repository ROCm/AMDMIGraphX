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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/rewrite_reshapes.hpp>
#include <migraphx/rewrite_broadcasts.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/fp8_types.hpp>
#include <migraphx/tune_axis.hpp>
#include <iterator>
#include <map>
#include <numeric>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_REDUCE_FUSION)

/// The names of the submodule parameters read as the indices of a gather
static std::unordered_set<std::string> gather_index_params(const module& sm)
{
    std::unordered_set<std::string> result;
    transform_if(
        sm.begin(),
        sm.end(),
        std::inserter(result, result.end()),
        [](const instruction& ins) {
            return ins.name() == "gather" and ins.inputs().back()->name() == "@param";
        },
        [](const instruction& ins) {
            return any_cast<builtin::param>(ins.inputs().back()->get_operator()).parameter;
        });
    return result;
}

struct fused_reduce
{
    std::vector<std::int64_t> axes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        const auto* sm = mods.front();
        auto outputs   = sm->get_output_shapes();
        if(outputs.empty())
            MIGRAPHX_THROW("fused_reduce: missing output");
        if(not sm->bypass())
            MIGRAPHX_THROW("fused_reduce: bypass flag is not set");
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this, true}.has(names.size());
        std::sort(names.begin(), names.end());
        auto shapes = sm->get_parameter_shapes();
        // Check dimension matches for each input
        if(not equal(names, inputs, [&](const auto& name, const auto& input) {
               auto s = shapes.at(name);
               return shape::same_lens(input, s);
           }))
            MIGRAPHX_THROW("Input dimension does not match the submodule.");
        // The indices of a gather in the submodule are not inputs of the
        // reduction, so they dont take part in the dimensions and the layout
        auto index_names = gather_index_params(*sm);
        std::vector<shape> tensors;
        auto is = range(names.size());
        transform_if(
            is.begin(),
            is.end(),
            std::back_inserter(tensors),
            [&](auto i) { return not contains(index_names, names[i]); },
            [&](auto i) { return inputs[i]; });
        check_shapes{tensors, *this, true}.same_ndims();

        if(outputs.front().dynamic())
            return outputs.size() == 1 ? outputs.front() : shape{outputs};

        // The output layout follows the inputs
        auto perm = find_permutation(tensors);
        std::vector<shape> result;
        std::transform(
            outputs.begin(), outputs.end(), std::back_inserter(result), [&](const shape& s) {
                return shape::from_permutation(s.type(), s.lens(), perm);
            });
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }

    std::string name() const { return "fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(fused_reduce);

/*
 * Predicate matcher checks that input and output shapes have the same rank.  This is assumed
 * for broadcast instructions for these fusions.
 */
MIGRAPHX_PRED_MATCHER(input_output_ndim_match, instruction_ref ins)
{
    auto input_shape  = ins->inputs().front()->get_shape();
    auto output_shape = ins->get_shape();
    return input_shape.ndim() == output_shape.ndim();
}

static auto
insert_module_in_submodule(module_ref sm,
                           instruction_ref ins,
                           std::unordered_map<instruction_ref, instruction_ref>* map_ins = nullptr,
                           module::inserter insert                                       = nullptr)
{
    assert(ins->module_inputs().size() == 1);
    return sm->fuse(*ins->module_inputs().front(), ins->inputs(), map_ins, std::move(insert));
}

static void create_reduce_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not ins->get_operator().attributes().get("reduce", false))
            continue;
        if(ins->inputs().size() != 1)
            continue;

        auto* rm =
            mpm.create_module(mpm.get_module().name() + ":" + ins->name() + std::to_string(n++));
        rm->set_bypass();

        rm->add_return(rm->fuse({ins}));
        auto v = ins->get_operator().to_value();

        // handle argmin/argmax
        std::vector<std::int64_t> axes;
        if(v.contains("axes"))
        {
            axes = v["axes"].to_vector<std::int64_t>();
        }
        else if(v.contains("axis"))
        {
            axes = {v["axis"].to<std::int64_t>()};
        }
        mpm.get_module().replace_instruction(
            ins, make_op("fused_reduce", {{"axes", axes}}), ins->inputs(), {rm});
    }
}

namespace {

instruction_ref get_broadcast_output(instruction_ref broadcast)
{
    if(broadcast->outputs().size() != 1)
        return broadcast;
    auto output = broadcast->outputs().front();
    if(output->name() == "contiguous")
        return get_broadcast_output(output);
    return output;
}

MIGRAPHX_PRED_MATCHER(used_once_except_broadcast, instruction_ref ins)
{
    if(ins->outputs().size() == 1)
        return true;
    if(ins->outputs().size() == 2)
    {
        auto is_broadcast = [](instruction_ref output) {
            return contains(output->name(), "broadcast");
        };
        auto broadcast = std::find_if(ins->outputs().begin(), ins->outputs().end(), is_broadcast);
        if(broadcast == ins->outputs().end())
            return false;
        auto non_broadcast =
            std::find_if_not(ins->outputs().begin(), ins->outputs().end(), is_broadcast);
        if(non_broadcast == ins->outputs().end())
            return false;
        auto output = get_broadcast_output(*broadcast);
        return output == *non_broadcast;
    }

    return false;
}
} // namespace
template <class... Ms>
static auto match_broadcast(Ms... ms)
{
    return match::skip(match::name("contiguous"))(
               match::name("multibroadcast", "broadcast")(
                   match::arg(0)(ms...), match::used_once(), input_output_ndim_match())
                   .bind("broadcast"))
        .bind("final_broadcast");
}

template <class... Ms>
static auto any_input(Ms... ms)
{
    return match::any_of[match::inputs()](match::any(ms...).bind("input"));
}

static bool is_valid_broadcast(const instruction_ref b, std::vector<size_t> reduce_axes)
{
    const auto& blens    = b->get_shape().lens();
    const auto& bstrides = b->get_shape().strides();
    reduce_axes.erase(std::remove_if(reduce_axes.begin(),
                                     reduce_axes.end(),
                                     [&](size_t axis) { return blens.at(axis) == 1; }),
                      reduce_axes.end());

    std::vector<size_t> broadcast_axes;
    copy_if(range(bstrides.size()), std::back_inserter(broadcast_axes), [&](size_t i) {
        return bstrides.at(i) == 0 and blens.at(i) != 1;
    });

    return broadcast_axes == reduce_axes;
}

template <class M>
static auto match_broadcast_axes(M m)
{
    return match::make_basic_fun_matcher(
        [=](match::matcher_context& ctx, instruction_ref ins) -> optional<instruction_ref> {
            optional<instruction_ref> result = m.match(ctx, ins);
            if(contains(ctx.instructions, "broadcast"))
            {
                instruction_ref reduce;
                if(ins->get_operator().name() == "fused_reduce")
                {
                    reduce = ins;
                }
                else
                {
                    assert(contains(ctx.instructions, "reduce"));
                    reduce = ctx.instructions["reduce"];
                }
                auto axes      = reduce->get_operator().to_value().at("axes").to_vector<size_t>();
                auto broadcast = ctx.instructions["broadcast"];
                if(not is_valid_broadcast(broadcast, axes))
                    return nullopt;
            }
            return result;
        });
}

static auto match_broadcastable_input(const std::string& op, const std::string& name)
{
    auto match_op                 = match::name(op)(used_once_except_broadcast()).bind(name);
    auto match_op_input           = any_input(match_op, match::used_once());
    auto broadcast_match_op_input = any_input(match_broadcast(match_op), match::used_once());
    return match::any_of(match_op_input, match_broadcast_axes(broadcast_match_op_input));
}

static void finalize_reduce_module(module_ref m)
{
    eliminate_common_subexpression{}.apply(*m);
    dead_code_elimination{}.apply(*m);
}

static std::vector<std::size_t> expand_dims(std::vector<std::size_t> lens,
                                            const std::vector<std::size_t>& axes,
                                            const std::vector<std::size_t>& dims)
{
    for(auto axis : axes)
        lens[axis] = dims[axis];
    return lens;
}

namespace {

bool has_unpack(instruction_ref ins)
{
    const auto* sm = ins->module_inputs().front();
    return std::any_of(
        sm->begin(), sm->end(), [](const auto& i) { return i.name() == "unpack_int4"; });
}

bool has_gather(instruction_ref ins)
{
    const auto* sm = ins->module_inputs().front();
    return std::any_of(sm->begin(), sm->end(), [](const auto& i) { return i.name() == "gather"; });
}

/// Whether the submodule reads its inputs through a layout the fusions cant
/// remap: packed inputs or gathered inputs
bool has_fixed_layout(instruction_ref ins) { return has_unpack(ins) or has_gather(ins); }

/// The inputs of the reduce that are reduced over, leaving out the indices
/// of the gathers in the submodule
std::vector<instruction_ref> reduce_tensor_inputs(instruction_ref reduce)
{
    const auto* sm   = reduce->module_inputs().front();
    auto index_names = gather_index_params(*sm);
    auto names       = sm->get_parameter_names();
    std::sort(names.begin(), names.end());
    assert(names.size() == reduce->inputs().size());
    std::vector<instruction_ref> result;
    auto is = range(names.size());
    transform_if(
        is.begin(),
        is.end(),
        std::back_inserter(result),
        [&](auto i) { return not contains(index_names, names[i]); },
        [&](auto i) { return reduce->inputs()[i]; });
    return result;
}

/// The axes of the reduce inputs gathered in the submodule
std::vector<std::size_t> gather_axes(instruction_ref reduce)
{
    const auto* sm = reduce->module_inputs().front();
    std::vector<std::size_t> result;
    transform_if(
        sm->begin(),
        sm->end(),
        std::back_inserter(result),
        [](const instruction& ins) { return ins.name() == "gather"; },
        [](const instruction& ins) {
            return tune_axis(ins.inputs().front()->get_shape().ndim(),
                             ins.get_operator().to_value().at("axis").to<int>(),
                             ins.name());
        });
    return result;
}

/// The lens with the axis split into (len / inner, inner); a unit axis stays unit
std::vector<std::size_t>
split_axis_lens(std::vector<std::size_t> lens, std::size_t axis, std::size_t inner = 2)
{
    assert(lens[axis] == 1 or lens[axis] % inner == 0);
    std::size_t n = lens[axis] == 1 ? 1 : inner;
    lens[axis] /= n;
    lens.insert(lens.begin() + axis + 1, n);
    return lens;
}

/// The lens of a broadcast input aligned to the broadcast output axes
std::vector<std::size_t> broadcast_input_lens(const shape& s)
{
    auto lens = s.lens();
    auto is   = range(lens.size());
    std::transform(is.begin(), is.end(), lens.begin(), [&](auto i) {
        return s.strides()[i] == 0 ? 1 : s.lens()[i];
    });
    return lens;
}

/// A view of the input with the axis split into (len / inner, inner): a
/// broadcast is rebuilt from its input so the reshape stays a view
optional<instruction_ref> insert_split_axis(
    module& m, instruction_ref pos, instruction_ref input, std::size_t axis, std::size_t inner = 2)
{
    const auto& s = input->get_shape();
    if(s.standard())
        return m.insert_instruction(
            pos, make_op("reshape", {{"dims", split_axis_lens(s.lens(), axis, inner)}}), input);
    if(not contains({"multibroadcast", "broadcast"}, input->name()))
        return nullopt;
    auto bin = input->inputs().front();
    if(not bin->get_shape().standard())
        return nullopt;
    auto lens = broadcast_input_lens(s);
    if(elements(lens) != bin->get_shape().elements())
        return nullopt;
    auto r = m.insert_instruction(
        pos, make_op("reshape", {{"dims", split_axis_lens(lens, axis, inner)}}), bin);
    return m.insert_instruction(
        pos, make_op("multibroadcast", {{"out_lens", split_axis_lens(s.lens(), axis, inner)}}), r);
}

std::size_t op_axis(const operation& op, std::size_t ndim)
{
    return tune_axis(ndim, op.to_value().at("axis").to<int>(), op.name());
}

/// A view of ins with the lens, as a squeeze or unsqueeze when the lens only
/// differ by unit dims since the reduce fusions see through those, otherwise
/// a reshape
instruction_ref insert_view_to_lens(module& m,
                                    instruction_ref pos,
                                    instruction_ref ins,
                                    const std::vector<std::size_t>& lens)
{
    const auto& src = ins->get_shape().lens();
    if(src == lens)
        return ins;
    std::vector<std::int64_t> squeezed;
    std::vector<std::int64_t> unsqueezed;
    std::size_t i = 0;
    std::size_t j = 0;
    while(i < src.size() or j < lens.size())
    {
        if(i < src.size() and j < lens.size() and src[i] == lens[j])
        {
            i++;
            j++;
        }
        else if(i < src.size() and src[i] == 1)
        {
            squeezed.push_back(i++);
        }
        else if(j < lens.size() and lens[j] == 1)
        {
            unsqueezed.push_back(j++);
        }
        else
        {
            break;
        }
    }
    if(i == src.size() and j == lens.size())
    {
        if(unsqueezed.empty())
            return m.insert_instruction(pos, make_op("squeeze", {{"axes", squeezed}}), ins);
        if(squeezed.empty())
            return m.insert_instruction(pos, make_op("unsqueeze", {{"axes", unsqueezed}}), ins);
    }
    return m.insert_instruction(pos, make_op("reshape", {{"dims", lens}}), ins);
}

/// The reduce axes once the axis is split in two: the axes after it move up
/// by one and a reduced split axis covers both parts
std::vector<std::int64_t> split_reduce_axes(const std::vector<std::size_t>& reduce_axes,
                                            std::size_t axis)
{
    std::vector<std::int64_t> axes;
    for(auto a : reduce_axes)
    {
        axes.push_back(a > axis ? a + 1 : a);
        if(a == axis)
            axes.push_back(a + 1);
    }
    return axes;
}

/// Rewrites the ops of a reduce submodule for inputs whose axis was split
/// into (len / inner, inner): the axes after it move up by one and the
/// broadcasts split the same way
struct split_axis_op
{
    std::size_t axis  = 0;
    std::size_t inner = 2;
    /// The rank of the inputs before the split
    std::size_t ndim = 0;
    /// The reduce axes after the split
    std::vector<std::int64_t> reduce_axes = {};

    std::int64_t remap_axis(std::size_t a) const { return a > axis ? a + 1 : a; }

    operation operator()(const operation& sop) const
    {
        auto v = sop.to_value();
        if(contains(sop.name(), "reduce"))
            return make_op(sop.name(), {{"axes", reduce_axes}});
        if(contains({"argmin", "argmax", "unpack_int4", "gather"}, sop.name()))
        {
            v["axis"] = remap_axis(op_axis(sop, ndim));
            return make_op(sop.name(), v);
        }
        if(contains({"multibroadcast", "broadcast"}, sop.name()))
        {
            auto out_lens = v.at("out_lens").to_vector<std::size_t>();
            return make_op("multibroadcast",
                           {{"out_lens", split_axis_lens(out_lens, axis, inner)}});
        }
        return sop;
    }
};

// Hoist a broadcast above a fused_reduce when it expands axes that were
// already size 1 on the reduce inputs rather than reduced axes. The leftover
// broadcast then only expands the reduced axes, which the fusion matchers
// can handle.
struct find_reduce_broadcast
{
    auto matcher() const
    {
        auto reduce = match::name("fused_reduce")(match::used_once()).bind("reduce");
        auto broadcast_reduce =
            match::name("multibroadcast")(
                match::args(reduce), match::nargs(1), match::used_once(), input_output_ndim_match())
                .bind("broadcast");
        return match::name("fused_reduce",
                           "pointwise")(match::any_of[match::inputs()](broadcast_reduce));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto broadcast = r.instructions["broadcast"];
        auto reduce    = r.instructions["reduce"];

        // The leftover broadcast can only fuse into a reduce over the same axes
        if(ins->name() == "fused_reduce" and ins->get_operator() != reduce->get_operator())
            return;
        // Rebuilding the submodule at expanded shapes cant remap packed or
        // gathered inputs
        if(has_fixed_layout(reduce))
            return;

        auto axes         = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        const auto& blens = broadcast->get_shape().lens();
        const auto& rlens = reduce->get_shape().lens();
        // Axes expanded by the broadcast which the reduce did not reduce
        std::vector<std::size_t> unreduced_axes;
        copy_if(range(blens.size()), std::back_inserter(unreduced_axes), [&](auto axis) {
            return blens[axis] != rlens[axis] and not contains(axes, axis);
        });
        if(unreduced_axes.empty())
            return;

        auto& m = mpm.get_module();
        std::vector<instruction_ref> new_inputs;
        std::transform(
            reduce->inputs().begin(),
            reduce->inputs().end(),
            std::back_inserter(new_inputs),
            [&](auto input) {
                auto out_lens = expand_dims(input->get_shape().lens(), unreduced_axes, blens);
                return m.insert_instruction(
                    broadcast, make_op("multibroadcast", {{"out_lens", out_lens}}), input);
            });

        // Rebuild the reduce module at the expanded shapes
        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + "_broadcast");
        rm->set_bypass();
        auto outs =
            rm->fuse(*old_rm,
                     new_inputs,
                     nullptr,
                     [&](module& rmm,
                         instruction_ref pos,
                         const operation& op,
                         const std::vector<instruction_ref>& inputs,
                         const std::vector<module_ref>& mod_args) {
                         if(contains({"multibroadcast", "broadcast"}, op.name()))
                         {
                             auto out_lens =
                                 expand_dims(op.to_value().at("out_lens").to_vector<std::size_t>(),
                                             unreduced_axes,
                                             blens);
                             return rmm.insert_instruction(
                                 pos, make_op("multibroadcast", {{"out_lens", out_lens}}), inputs);
                         }
                         return rmm.insert_instruction(pos, op, inputs, mod_args);
                     });
        rm->add_return(outs);

        auto new_reduce = m.insert_instruction(broadcast, reduce->get_operator(), new_inputs, {rm});
        if(new_reduce->get_shape().lens() == blens)
            m.replace_instruction(broadcast, new_reduce);
        else
            m.replace_instruction(
                broadcast, make_op("multibroadcast", {{"out_lens", blens}}), new_reduce);
    }
};

struct find_pointwise_reduce
{
    auto matcher() const
    {
        // fused_reduce instruction with pointwise inputs.
        return match::name("fused_reduce")(match_broadcastable_input("pointwise", "pointwise"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce        = r.result;
        auto input         = r.instructions["pointwise"];
        const auto* pm     = input->module_inputs().front();
        const auto* old_rm = reduce->module_inputs().front();

        auto* rm = mpm.create_module(pm->name() + ":" + old_rm->name());
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Insert pointwise
        auto rins      = rm->fuse({input}, &map_ins).front();
        map_ins[input] = rins;

        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast     = r.instructions["broadcast"];
            auto fbroadcast    = r.instructions["final_broadcast"];
            map_ins[broadcast] = rm->fuse({broadcast}, &map_ins).front();
            if(fbroadcast != broadcast)
                map_ins[fbroadcast] = map_ins[broadcast];
        }

        // Insert fused_reduce
        rm->add_return(insert_module_in_submodule(rm, reduce, &map_ins));
        finalize_reduce_module(rm);

        auto new_inputs = find_inputs(map_ins, &mpm.get_module(), rm);
        mpm.get_module().replace_instruction(reduce, reduce->get_operator(), new_inputs, {rm});
    }
};

struct find_reduce_pointwise
{

    auto matcher() const
    {
        return match::name("pointwise")(match_broadcastable_input("fused_reduce", "reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto pw     = r.result;
        auto reduce = r.instructions["reduce"];
        auto input  = r.instructions["input"];

        const auto* pm     = pw->module_inputs().front();
        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":" + pm->name());
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Copy module instructions
        insert_module_in_submodule(rm, reduce, &map_ins);
        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast                       = r.instructions["broadcast"];
            map_ins[broadcast->inputs().front()] = rm->get_returns().front();
            auto bout                            = rm->fuse({broadcast}, &map_ins);
            map_ins[input]                       = bout.front();
        }
        else
        {
            map_ins[input] = rm->get_returns().front();
        }

        auto out = rm->fuse({pw}, &map_ins);
        rm->replace_return(out);
        finalize_reduce_module(rm);

        auto new_inputs = find_inputs(map_ins, &mpm.get_module(), rm);
        mpm.get_module().replace_instruction(pw, reduce->get_operator(), new_inputs, {rm});
    }
};

struct find_reduce_reduce
{
    auto matcher() const
    {
        return match::name("fused_reduce")(match_broadcastable_input("fused_reduce", "reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce1 = r.result;
        auto reduce2 = r.instructions["reduce"];
        auto input   = r.instructions["input"];

        if(reduce1->get_operator() != reduce2->get_operator())
            return;

        const auto* rm1 = reduce1->module_inputs().front();
        const auto* rm2 = reduce2->module_inputs().front();
        auto* rm        = mpm.create_module(rm1->name() + ":" + rm2->name());
        rm->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        // Copy reduce1 instructions
        insert_module_in_submodule(rm, reduce2, &map_ins);
        if(contains(r.instructions, "broadcast"))
        {
            auto broadcast                       = r.instructions["broadcast"];
            map_ins[broadcast->inputs().front()] = rm->get_returns().front();
            auto bout                            = rm->fuse({broadcast}, &map_ins);
            map_ins[input]                       = bout.front();
        }
        else
        {
            map_ins[input] = rm->get_returns().front();
        }

        auto out = insert_module_in_submodule(rm, reduce1, &map_ins);
        rm->replace_return(out);
        finalize_reduce_module(rm);

        auto new_inputs = find_inputs(map_ins, &mpm.get_module(), rm);
        mpm.get_module().replace_instruction(reduce1, reduce1->get_operator(), new_inputs, {rm});
    }
};

// Fuse an unpack_int4 feeding a fused_reduce into the submodule so the
// packed data is read directly by the reduction kernel. The kernel unpacks
// by vectorizing the packed input with half the vector size, so only fuse
// when every input can be vectorized along the unpack axis.
struct find_unpack_reduce
{
    auto matcher() const
    {
        auto unpack = match::name("unpack_int4")(match::used_once()).bind("unpack");
        auto reshapes =
            match::name("reshape", "squeeze", "unsqueeze", "flatten")(match::used_once());
        return match::name("fused_reduce")(
            any_input(match::skip(reshapes)(unpack), match::used_once()));
    }

    static std::size_t normalized_axis(instruction_ref unpack)
    {
        return tune_axis(unpack->get_shape().ndim(),
                         unpack->get_operator().to_value().at("axis").to<int>(),
                         unpack->name());
    }

    // Push the unpack past the reshapes so it feeds the reduce directly,
    // reshaping the packed input instead so it can be fused
    static optional<instruction_ref>
    hoist_unpack(module& m, instruction_ref input, instruction_ref unpack)
    {
        if(input == unpack)
            return unpack;
        std::vector<operation> ops;
        auto next_ins = input;
        while(next_ins != unpack)
        {
            ops.push_back(next_ins->get_operator());
            next_ins = next_ins->inputs().front();
        }
        std::reverse(ops.begin(), ops.end());
        auto desc = shape_transform_descriptor::create(unpack->get_shape().lens(), ops);
        if(desc.empty() or desc.has_broadcast())
            return nullopt;
        auto axes = desc.get_dst_axes_from_src(normalized_axis(unpack));
        if(axes.empty())
            return nullopt;
        auto axis = axes.back();
        auto lens = input->get_shape().lens();
        if(lens[axis] % 2 != 0)
            return nullopt;
        lens[axis] /= 2;
        auto packed = unpack->inputs().front();
        if(elements(lens) != packed->get_shape().elements())
            return nullopt;
        auto packed_reshape =
            m.insert_instruction(input, make_op("reshape", {{"dims", lens}}), packed);
        auto new_unpack =
            m.insert_instruction(input, make_op("unpack_int4", {{"axis", axis}}), packed_reshape);
        return m.replace_instruction(input, new_unpack);
    }

    static bool is_vectorizable_by_two(const shape& s, std::size_t axis)
    {
        if(s.lens()[axis] != 1)
        {
            if(s.strides()[axis] > 1)
                return false;
            if(s.strides()[axis] == 1 and s.lens()[axis] % 2 != 0)
                return false;
        }
        return std::all_of(s.strides().begin(), s.strides().end(), [](auto stride) {
            return stride < 2 or stride % 2 == 0;
        });
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce  = r.result;
        auto input   = r.instructions["input"];
        auto hoisted = hoist_unpack(mpm.get_module(), input, r.instructions["unpack"]);
        if(not hoisted.has_value())
            return;
        auto unpack = *hoisted;
        auto axis   = normalized_axis(unpack);
        auto packed = unpack->inputs().front();
        // The unpack axis must be the fastest dimension to read the packed
        // data vectorized
        if(packed->get_shape().strides()[axis] != 1)
            return;
        // Vectorization requires the unpack axis to be reduced
        auto axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        if(not contains(axes, axis))
            return;
        static const auto fp8 = fp8_types{}.get();
        if(not all_of(reduce_tensor_inputs(reduce), [&](auto ri) {
               if(contains(fp8, ri->get_shape().type()))
                   return false;
               return is_vectorizable_by_two(ri->get_shape(), axis);
           }))
            return;

        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":unpack_int4");
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        map_ins[unpack] = rm->fuse({unpack}, &map_ins).front();
        rm->add_return(insert_module_in_submodule(rm, reduce, &map_ins));
        finalize_reduce_module(rm);

        auto new_inputs = find_inputs(map_ins, &mpm.get_module(), rm);
        mpm.get_module().replace_instruction(reduce, reduce->get_operator(), new_inputs, {rm});
    }
};

struct reduce_reshape : rewrite_reshapes_base
{
    static std::string name() { return "fused_reduce"; }

    static bool matches(instruction_ref ins)
    {
        if(ins->name() != name())
            return true;
        // Submodules with packed or gathered inputs cant be remapped to the
        // common dims
        return not has_fixed_layout(ins);
    }

    template <class Transform>
    static auto transform_op(const Transform& t)
    {
        return [=](module& m,
                   instruction_ref ins,
                   const operation& op,
                   const std::vector<instruction_ref>& inputs,
                   const std::vector<module_ref>& mod_args) {
            auto new_op = t(op);
            return m.insert_instruction(ins, new_op, inputs, mod_args);
        };
    }

    template <class AxesMap>
    static instruction_ref insert(module_pass_manager& mpm,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& inputs,
                                  const AxesMap& am)
    {
        auto op = any_cast<fused_reduce>(ins->get_operator());
        std::vector<int64_t> axes;
        for(auto axis : op.axes)
        {
            auto new_axes = am.at(axis);
            axes.insert(axes.end(), new_axes.begin(), new_axes.end());
        }
        std::sort(axes.begin(), axes.end());
        auto dims  = base_dims(inputs);
        auto* oldm = ins->module_inputs().front();
        auto* sm   = mpm.create_module(oldm->name() + "_reshape");
        sm->set_bypass();
        auto outs = sm->fuse(*oldm, inputs, nullptr, transform_op([&](const operation& sop) {
            if(contains(sop.name(), "reduce"))
                return make_op(sop.name(), {{"axes", axes}});
            // handle argmin/argmax
            if(sop.name() == "argmin" or sop.name() == "argmax")
            {
                auto v    = sop.to_value();
                v["axis"] = axes.front();
                return make_op(sop.name(), v);
            }
            if(contains({"multibroadcast", "broadcast"}, sop.name()))
                return make_op("multibroadcast", {{"out_lens", dims}});
            assert(sop.name() == "pointwise");
            return sop;
        }));
        sm->add_return(outs);
        return mpm.get_module().insert_instruction(ins, fused_reduce{axes}, inputs, {sm});
    }

    static std::vector<std::size_t> base_dims(const std::vector<instruction_ref>& inputs)
    {
        auto input = std::max_element(inputs.begin(), inputs.end(), by(std::less<>{}, [](auto i) {
                                          return i->get_shape().elements();
                                      }));
        return (*input)->get_shape().lens();
    }

    static std::vector<std::size_t> base_dims(instruction_ref ins)
    {
        return base_dims(ins->inputs());
    }
};

/// Whether an input of the reduce is an unpack_int4 that has not been
/// fused into the reduce yet
bool input_has_unpack(instruction_ref reduce)
{
    static const std::unordered_set<std::string> view_ops = {"reshape",
                                                             "squeeze",
                                                             "unsqueeze",
                                                             "flatten",
                                                             "transpose",
                                                             "multibroadcast",
                                                             "broadcast",
                                                             "contiguous"};
    return any_of(reduce->inputs(), [&](instruction_ref input) {
        while(input->inputs().size() == 1 and contains(view_ops, input->name()))
            input = input->inputs().front();
        return input->name() == "unpack_int4";
    });
}

/// Split a fused_reduce that is only consumed by slices along a non-reduced
/// axis into one reduce per slice over the sliced inputs, so the consumers
/// of the slices can fuse with the reductions (eg swiglu over the halves of
/// a gate_up matvec). When the slices cut one part of a reduce axis that a
/// reshape split in two, eg interleaved gate and up columns, the axis of the
/// reduce is split the same way first. Waits for the unpack to be fused
/// since the slices are pushed into the reduce inputs.
struct find_reduce_slice
{
    auto matcher() const
    {
        auto reshapes = match::name("reshape", "squeeze", "unsqueeze", "transpose");
        return match::name("slice")(
            match::arg(0)(match::skip(reshapes)(match::name("fused_reduce").bind("reduce"))));
    }

    static std::vector<std::size_t> slice_axes(instruction_ref slice)
    {
        return slice->get_operator().to_value().at("axes").to_vector<std::size_t>();
    }

    /// How the sliced axis maps onto the reduce output through the views
    /// between them
    struct sliced_axis
    {
        /// The reduce axis
        std::size_t axis = 0;
        /// The inner length the reduce axis is split into when the slice
        /// cuts one part of it, 1 when it cuts the whole axis
        std::size_t inner = 1;
        /// The views from the reduce to the slice
        std::vector<operation> ops = {};
    };

    static optional<sliced_axis>
    find_sliced_axis(instruction_ref input, instruction_ref reduce, std::size_t axis)
    {
        std::vector<operation> ops;
        for(auto ins = input; ins != reduce; ins = ins->inputs().front())
        {
            if(ins != input and ins->outputs().size() != 1)
                return nullopt;
            ops.push_back(ins->get_operator());
        }
        if(input != reduce and reduce->outputs().size() != 1)
            return nullopt;
        std::reverse(ops.begin(), ops.end());
        if(ops.empty())
            return sliced_axis{axis, 1, ops};
        // A reshape in the chain is replayed as a reshape to the sliced lens,
        // which a transpose would reorder
        bool has_reshape = any_of(ops, [](const operation& op) { return op.name() == "reshape"; });
        if(has_reshape and
           any_of(ops, [](const operation& op) { return op.name() == "transpose"; }))
            return nullopt;
        const auto& rlens = reduce->get_shape().lens();
        const auto& lens  = input->get_shape().lens();
        auto desc         = shape_transform_descriptor::create(rlens, ops);
        if(desc.empty())
            return nullopt;
        for(std::size_t i : range(rlens.size()))
        {
            // The unit dims of the views are attributed to the axes next to them
            auto dst = desc.get_dst_axes_from_src(i);
            dst.erase(std::remove_if(dst.begin(), dst.end(), [&](auto a) { return lens[a] == 1; }),
                      dst.end());
            if(not contains(dst, axis))
                continue;
            if(dst.size() == 1)
                return sliced_axis{i, 1, ops};
            // A reshape split the axis in two and the slice cuts the inner part
            if(dst.size() != 2 or dst.back() != axis or lens[dst[0]] * lens[dst[1]] != rlens[i])
                return nullopt;
            return sliced_axis{i, lens[axis], ops};
        }
        return nullopt;
    }

    /// Split the non-reduced axis of the reduce into (len / inner, inner) on
    /// every input and in the submodule. A consumer reshaping the reduce to
    /// the split lens reads the split reduce directly, otherwise the result
    /// is reshaped back to the reduce lens.
    static optional<instruction_ref> split_reduce_axis(module_pass_manager& mpm,
                                                       instruction_ref reduce,
                                                       std::size_t axis,
                                                       std::size_t inner)
    {
        auto& m = mpm.get_module();
        if(reduce->get_shape().type() == shape::tuple_type)
            return nullopt;
        auto tensors = reduce_tensor_inputs(reduce);
        std::vector<instruction_ref> inputs;
        for(auto input : reduce->inputs())
        {
            // Gather indices are not reduce inputs
            if(not contains(tensors, input))
            {
                inputs.push_back(input);
                continue;
            }
            auto split = insert_split_axis(m, reduce, input, axis, inner);
            if(not split.has_value())
                return nullopt;
            inputs.push_back(*split);
        }
        auto rlens       = reduce->get_shape().lens();
        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        split_axis_op rewrite{axis, inner, rlens.size(), split_reduce_axes(reduce_axes, axis)};
        const auto* oldm = reduce->module_inputs().front();
        auto* sm         = mpm.create_module(oldm->name() + "_split");
        sm->set_bypass();
        auto outs = sm->fuse(*oldm, inputs, nullptr, reduce_reshape::transform_op(rewrite));
        sm->add_return(outs);
        finalize_reduce_module(sm);
        auto new_reduce =
            m.insert_instruction(reduce, fused_reduce{rewrite.reduce_axes}, inputs, {sm});
        auto consumer = reduce->outputs().front();
        if(consumer->name() == "reshape")
            m.replace_instruction(
                consumer,
                insert_view_to_lens(m, consumer, new_reduce, consumer->get_shape().lens()));
        else
            m.replace_instruction(reduce, make_op("reshape", {{"dims", rlens}}), new_reduce);
        return new_reduce;
    }

    /// The views between the reduce and the slice applied to the sliced
    /// reduce: unit reshapes replay as they are, but a reshape to fixed dims
    /// cant, so such a chain becomes a view to the lens of the slice
    static instruction_ref replay_views(module& m,
                                        instruction_ref slice,
                                        instruction_ref new_reduce,
                                        const std::vector<operation>& ops)
    {
        if(none_of(ops, [](const operation& op) { return op.name() == "reshape"; }))
        {
            return std::accumulate(
                ops.begin(), ops.end(), new_reduce, [&](instruction_ref ins, const operation& op) {
                    return m.insert_instruction(slice, op, ins);
                });
        }
        return insert_view_to_lens(m, slice, new_reduce, slice->get_shape().lens());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m     = mpm.get_module();
        auto slice  = r.result;
        auto reduce = r.instructions["reduce"];
        if(input_has_unpack(reduce))
            return;
        auto axes = slice_axes(slice);
        if(axes.size() != 1)
            return;
        auto v     = slice->get_operator().to_value();
        auto start = v.at("starts").to_vector<std::int64_t>().front();
        auto end   = v.at("ends").to_vector<std::int64_t>().front();
        auto input = slice->inputs().front();
        // Every consumer of the reduce must be a slice along the same axis,
        // otherwise the reduction would be duplicated
        if(not all_of(input->outputs(), [&](instruction_ref out) {
               return out->name() == "slice" and slice_axes(out) == axes;
           }))
            return;
        auto sliced = find_sliced_axis(input, reduce, axes.front());
        if(not sliced.has_value())
            return;
        auto raxis       = sliced->axis;
        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        if(contains(reduce_axes, raxis))
            return;
        // Slicing the gathered axis would slice the data rows, not the indices
        if(contains(gather_axes(reduce), raxis))
            return;
        if(sliced->inner > 1)
        {
            // The slice then cuts a whole axis of the split reduce, which is
            // sliced right away so the other slices of this sweep see the
            // same structure
            auto split = split_reduce_axis(mpm, reduce, raxis, sliced->inner);
            if(not split.has_value())
                return;
            reduce = *split;
            // The split replaced the view the slice read
            input  = slice->inputs().front();
            sliced = find_sliced_axis(input, reduce, axes.front());
            if(not sliced.has_value() or sliced->inner > 1)
                return;
            raxis = sliced->axis;
        }
        const auto& rlens = reduce->get_shape().lens();
        std::int64_t len  = rlens[raxis];
        if(start < 0 or end <= start or end > len or end - start == len)
            return;
        auto tensors = reduce_tensor_inputs(reduce);
        if(not all_of(tensors, [&](instruction_ref x) {
               return x->get_shape().lens()[raxis] == rlens[raxis];
           }))
            return;
        auto slice_op = make_op("slice", {{"axes", {raxis}}, {"starts", {start}}, {"ends", {end}}});
        auto inputs   = reduce->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](instruction_ref x) {
            if(not contains(tensors, x))
                return x;
            return m.insert_instruction(slice, slice_op, x);
        });
        // Broadcasts inside the submodule expand to the full axis
        std::size_t new_len = end - start;
        const auto* oldm    = reduce->module_inputs().front();
        auto* sm            = mpm.create_module(oldm->name() + "_slice" + std::to_string(start));
        sm->set_bypass();
        auto outs = sm->fuse(
            *oldm, inputs, nullptr, reduce_reshape::transform_op([&](const operation& sop) {
                if(not contains({"multibroadcast", "broadcast"}, sop.name()))
                    return sop;
                auto sv       = sop.to_value();
                auto out_lens = sv.at("out_lens").to_vector<std::size_t>();
                if(raxis < out_lens.size() and out_lens[raxis] == rlens[raxis])
                    out_lens[raxis] = new_len;
                sv["out_lens"] = out_lens;
                return make_op(sop.name(), sv);
            }));
        sm->add_return(outs);
        auto new_reduce = m.insert_instruction(slice, reduce->get_operator(), inputs, {sm});
        auto y          = replay_views(m, slice, new_reduce, sliced->ops);
        assert(y->get_shape().lens() == slice->get_shape().lens());
        m.replace_instruction(slice, y);
    }
};

/// Fuse a gather along a non-reduced axis feeding a fused_reduce into the
/// submodule, so the kernel reads the gathered rows of the data in place
/// instead of copying them first, eg the weights of the selected experts of
/// a MoE layer. The data takes the place of the gather in the chain of views
/// to the reduce, with the gather axis at the data length, and the gather
/// moves inside reading the data view and the indices.
struct find_gather_reduce
{
    /// The views the gather can be moved through
    static const std::unordered_set<std::string>& view_names()
    {
        static const std::unordered_set<std::string> names = {
            "reshape", "squeeze", "unsqueeze", "flatten", "multibroadcast", "broadcast", "slice"};
        return names;
    }

    auto matcher() const
    {
        auto gather = match::name("gather")(match::nargs(2));
        auto views  = match::name(view_names());
        return match::name("fused_reduce")(
            any_input(match::skip(views)(gather), match::used_once()));
    }

    /// The gather at the end of the chain of views from the input, if the
    /// input is a view of a gather of the shapes the kernel can read. The
    /// views and the gather may feed other inputs as well, eg the two slices
    /// of a split reduce.
    static optional<instruction_ref> find_gather(instruction_ref input)
    {
        if(input->outputs().size() != 1)
            return nullopt;
        auto ins = input;
        while(ins->name() != "gather")
        {
            if(ins->inputs().size() != 1)
                return nullopt;
            if(not contains(view_names(), ins->name()))
                return nullopt;
            ins = ins->inputs().front();
        }
        if(ins->inputs().size() != 2)
            return nullopt;
        const auto& ishape = ins->inputs().back()->get_shape();
        // The kernel reads the indices as a vector, and the gather axis must
        // keep more than one element so it survives the merging of the
        // dimensions in the kernel
        if(ishape.dynamic() or ishape.ndim() != 1 or ishape.elements() < 2)
            return nullopt;
        if(ins->inputs().front()->get_shape().dynamic())
            return nullopt;
        return ins;
    }

    /// The axis the gather axis maps to through the view, when it stays a
    /// whole axis; a slice keeps the axes
    static optional<std::size_t> map_axis(instruction_ref view, std::size_t axis)
    {
        if(view->name() == "slice")
            return axis;
        const auto& in_lens = view->inputs().front()->get_shape().lens();
        auto desc           = shape_transform_descriptor::create(in_lens, {view->get_operator()});
        if(desc.empty())
            return nullopt;
        // The unit dims of the view are attributed to the axes next to them
        const auto& lens = view->get_shape().lens();
        auto axes        = desc.get_dst_axes_from_src(axis);
        axes.erase(std::remove_if(axes.begin(), axes.end(), [&](auto a) { return lens[a] == 1; }),
                   axes.end());
        if(axes.size() != 1)
            return nullopt;
        return axes.front();
    }

    /// The view applied to the data in place of the gather output: the
    /// gather axis has the data length, and a slice must not cut it
    static optional<operation>
    data_view_op(instruction_ref view, std::size_t axis, std::size_t data_len)
    {
        auto op = view->get_operator();
        auto v  = op.to_value();
        if(op.name() == "slice")
        {
            if(contains(v.at("axes").to_vector<std::size_t>(), axis))
                return nullopt;
            return op;
        }
        if(op.name() == "reshape")
        {
            auto dims  = v.at("dims").to_vector<std::int64_t>();
            dims[axis] = data_len;
            return make_op("reshape", {{"dims", dims}});
        }
        if(contains({"multibroadcast", "broadcast"}, op.name()))
        {
            auto out_lens  = v.at("out_lens").to_vector<std::size_t>();
            out_lens[axis] = data_len;
            v["out_lens"]  = out_lens;
            return make_op(op.name(), v);
        }
        return op;
    }

    /// Insert a gather of the data viewed like the input in place of the
    /// input: the views between the gather and the input are replayed on the
    /// data with the gather axis at the data length
    static optional<instruction_ref> insert_gather_view(module& m,
                                                        instruction_ref reduce,
                                                        instruction_ref input,
                                                        instruction_ref gather)
    {
        auto data     = gather->inputs().front();
        auto indices  = gather->inputs().back();
        auto axis     = tune_axis(data->get_shape().ndim(),
                                  gather->get_operator().to_value().at("axis").to<int>(),
                                  gather->name());
        auto data_len = data->get_shape().lens()[axis];
        std::vector<instruction_ref> views;
        for(auto ins = input; ins != gather; ins = ins->inputs().front())
            views.push_back(ins);
        std::reverse(views.begin(), views.end());
        auto view = data;
        for(auto ins : views)
        {
            auto mapped = map_axis(ins, axis);
            if(not mapped.has_value())
                return nullopt;
            auto op = data_view_op(ins, *mapped, data_len);
            if(not op.has_value())
                return nullopt;
            view = m.insert_instruction(reduce, *op, view);
            axis = *mapped;
        }
        const auto& s = input->get_shape();
        auto lens     = s.lens();
        lens[axis]    = data_len;
        if(view->get_shape().lens() != lens)
            return nullopt;
        // The gather axis must reach the reduce whole, not broadcast and not reduced
        if(s.lens()[axis] != indices->get_shape().elements() or s.strides()[axis] == 0)
            return nullopt;
        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        if(contains(reduce_axes, axis))
            return nullopt;
        return m.insert_instruction(reduce, make_op("gather", {{"axis", axis}}), view, indices);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m     = mpm.get_module();
        auto reduce = r.result;
        // Every gathered input moves in at once so the submodule is rebuilt once
        std::vector<instruction_ref> gathers;
        std::vector<instruction_ref> inputs;
        for(auto input : reduce->inputs())
        {
            auto gather = find_gather(input);
            if(not gather.has_value())
                continue;
            auto gathered = insert_gather_view(m, reduce, input, *gather);
            if(not gathered.has_value())
                continue;
            gathers.push_back(*gathered);
            inputs.push_back(input);
        }
        if(gathers.empty())
            return;

        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":gather");
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        rm->fuse(gathers, &map_ins);
        // The inputs read the fused gathers
        std::transform(inputs.begin(),
                       inputs.end(),
                       gathers.begin(),
                       std::inserter(map_ins, map_ins.end()),
                       [&](instruction_ref input, instruction_ref gathered) {
                           return std::make_pair(input, map_ins.at(gathered));
                       });
        rm->add_return(insert_module_in_submodule(rm, reduce, &map_ins));
        finalize_reduce_module(rm);

        auto new_inputs = find_inputs(map_ins, &m, rm);
        m.replace_instruction(reduce, reduce->get_operator(), new_inputs, {rm});
    }
};

/// Map a pointwise over squeezed fused_reduce outputs into the reduce space
/// so it can fuse as an epilogue of the reductions: the other inputs are
/// unsqueezed instead and the result is squeezed after the pointwise
struct find_reduce_squeeze_pointwise
{
    auto matcher() const
    {
        auto squeeze =
            match::name("squeeze")(match::used_once(), match::arg(0)(match::name("fused_reduce")));
        return match::name("pointwise")(match::any_of[match::inputs()](squeeze.bind("squeeze")));
    }

    static std::vector<std::int64_t> squeeze_axes(instruction_ref squeeze)
    {
        return squeeze->get_operator().to_value().at("axes").to_vector<std::int64_t>();
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m   = mpm.get_module();
        auto pw   = r.result;
        auto axes = squeeze_axes(r.instructions["squeeze"]);
        if(axes.empty())
            return;
        if(pw->get_shape().type() == shape::tuple_type)
            return;
        auto is_squeezed_reduce = [&](instruction_ref input) {
            if(input->name() != "squeeze" or input->outputs().size() != 1)
                return false;
            if(input->inputs().front()->name() != "fused_reduce")
                return false;
            return squeeze_axes(input) == axes;
        };
        const auto& rlens = r.instructions["squeeze"]->inputs().front()->get_shape().lens();
        auto inputs       = pw->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](instruction_ref input) {
            if(is_squeezed_reduce(input))
                return input->inputs().front();
            // A {1} input cant be unsqueezed back to the reduce shape: squeezing all
            // dims clamps the rank at 1 and unsqueeze is a no-op on scalars. The
            // reduce shape is all ones then, so broadcast instead.
            if(input->get_shape().lens() == std::vector<std::size_t>{1})
                return m.insert_instruction(
                    pw, make_op("multibroadcast", {{"out_lens", rlens}}), input);
            return m.insert_instruction(pw, make_op("unsqueeze", {{"axes", axes}}), input);
        });
        auto new_pw = m.insert_instruction(pw, pw->get_operator(), inputs, pw->module_inputs());
        m.replace_instruction(pw, make_op("squeeze", {{"axes", axes}}), new_pw);
    }
};

// Fuse an unpack_int4 that reaches a fused_reduce through a broadcast over a
// faster reduced axis, such as a per-block zero point broadcast over the
// block elements. The kernel reads a packed input by vectorizing along the
// unpack axis, which is not the vectorized axis here, so instead the unpack
// axis is split in two: the packed bytes become a view broadcast over both
// nibbles and a pointwise selects the nibble with a per-nibble literal.
struct find_unpack_broadcast_reduce
{
    auto matcher() const
    {
        auto unpack = match::name("unpack_int4")(match::used_once()).bind("unpack");
        auto reshapes =
            match::name("reshape", "squeeze", "unsqueeze", "flatten")(match::used_once());
        auto broadcast = match::name("multibroadcast", "broadcast")(
                             match::used_once(), match::arg(0)(match::skip(reshapes)(unpack)))
                             .bind("broadcast");
        return match::name("fused_reduce")(match::any_of[match::inputs()](broadcast));
    }

    /// The reduce input axis the unpack axis maps to through the chain of
    /// reshapes and the broadcast, if it maps to a whole axis
    static optional<std::size_t> find_unpack_axis(instruction_ref broadcast, instruction_ref unpack)
    {
        std::vector<operation> ops;
        for(auto ins = broadcast; ins != unpack; ins = ins->inputs().front())
            ops.push_back(ins->get_operator());
        std::reverse(ops.begin(), ops.end());
        auto desc = shape_transform_descriptor::create(unpack->get_shape().lens(), ops);
        if(desc.empty())
            return nullopt;
        auto uaxis = op_axis(unpack->get_operator(), unpack->get_shape().ndim());
        auto axes  = desc.get_dst_axes_from_src(uaxis);
        if(axes.size() != 1)
            return nullopt;
        auto axis = axes.front();
        if(broadcast->get_shape().lens()[axis] != unpack->get_shape().lens()[uaxis])
            return nullopt;
        return axis;
    }

    /// The axis to split, when the packed pairs are adjacent along a reduced
    /// axis that is broadcast over a faster one, so the vectorized unpack of
    /// find_unpack_reduce does not apply, and every input can be split
    static optional<std::size_t>
    find_split_axis(instruction_ref reduce, instruction_ref broadcast, instruction_ref unpack)
    {
        auto packed = unpack->inputs().front();
        if(packed->get_shape().type() != shape::uint8_type or not packed->get_shape().standard())
            return nullopt;
        auto axis = find_unpack_axis(broadcast, unpack);
        if(not axis.has_value())
            return nullopt;
        const auto& bshape = broadcast->get_shape();
        if(bshape.strides()[*axis] != 1 or bshape.lens()[*axis] % 2 != 0)
            return nullopt;
        if(std::none_of(bshape.lens().begin() + *axis + 1, bshape.lens().end(), [](auto len) {
               return len > 1;
           }))
            return nullopt;
        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        if(not contains(reduce_axes, *axis))
            return nullopt;
        // An epilogue input at the output shape is unit along the axis
        if(not all_of(reduce_tensor_inputs(reduce), [&](instruction_ref input) {
               auto len = input->get_shape().lens()[*axis];
               return len == bshape.lens()[*axis] or len == 1;
           }))
            return nullopt;
        const auto* rm = reduce->module_inputs().front();
        if(any_of(*rm, [&](const instruction& ins) {
               return ins.name() == "unpack_int4" and
                      op_axis(ins.get_operator(), bshape.ndim()) == *axis;
           }))
            return nullopt;
        auto blens = broadcast_input_lens(bshape);
        blens[*axis] /= 2;
        if(elements(blens) != packed->get_shape().elements())
            return nullopt;
        return axis;
    }

    /// The packed bytes as a view broadcast over both nibbles of the split
    /// axis, and the nibble select literal along the nibble axis
    static std::pair<instruction_ref, instruction_ref>
    insert_packed_inputs(module& m,
                         instruction_ref reduce,
                         instruction_ref broadcast,
                         instruction_ref packed,
                         std::size_t axis)
    {
        auto dims  = split_axis_lens(broadcast->get_shape().lens(), axis);
        auto blens = broadcast_input_lens(broadcast->get_shape());
        blens[axis] /= 2;
        blens.insert(blens.begin() + axis + 1, 1);
        auto bytes = m.insert_instruction(reduce, make_op("reshape", {{"dims", blens}}), packed);
        bytes =
            m.insert_instruction(reduce, make_op("multibroadcast", {{"out_lens", dims}}), bytes);
        // Select 16 reads the low nibble and 1 the high nibble
        std::vector<std::size_t> select_lens(dims.size(), 1);
        select_lens[axis + 1] = 2;
        auto select = m.insert_literal(reduce, literal{shape{shape::uint8_type, {2}}, {16, 1}});
        select = m.insert_instruction(reduce, make_op("reshape", {{"dims", select_lens}}), select);
        select =
            m.insert_instruction(reduce, make_op("multibroadcast", {{"out_lens", dims}}), select);
        return {bytes, select};
    }

    /// Select nibble x0 of byte x1: the low nibble with select 16 (shifted up
    /// then down) and the high nibble with select 1
    static module_ref create_nibble_module(module_pass_manager& mpm, const std::string& name)
    {
        auto* pm = mpm.create_module(name);
        pm->set_bypass();
        shape s{shape::uint8_type};
        auto byte    = pm->add_parameter("x0", s);
        auto select  = pm->add_parameter("x1", s);
        auto sixteen = pm->add_literal(literal{s, {16}});
        auto fifteen = pm->add_literal(literal{s, {15}});
        auto shifted = pm->add_instruction(make_op("mul"), byte, select);
        auto high    = pm->add_instruction(make_op("div"), shifted, sixteen);
        auto nibble  = pm->add_instruction(make_op("bitwise_and"), high, fifteen);
        pm->add_return({nibble});
        return pm;
    }

    /// The reduce submodule at the split dims, with the nibble select in
    /// place of the broadcast unpack parameter
    static module_ref
    create_split_module(module_pass_manager& mpm,
                        instruction_ref reduce,
                        instruction_ref broadcast,
                        const std::vector<instruction_ref>& inputs,
                        std::size_t axis,
                        const std::vector<std::int64_t>& axes,
                        std::unordered_map<instruction_ref, instruction_ref>& map_ins)
    {
        const auto* oldm = reduce->module_inputs().front();
        auto* sm         = mpm.create_module(oldm->name() + ":unpack_int4");
        sm->set_bypass();
        sm->add_params(inputs, &map_ins);
        // The packed bytes are in place of the broadcast, the select is last
        auto bit    = std::find(reduce->inputs().begin(), reduce->inputs().end(), broadcast);
        auto bytes  = inputs[std::distance(reduce->inputs().begin(), bit)];
        auto* pm    = create_nibble_module(mpm, sm->name() + ":nibble");
        auto nibble = sm->add_instruction(
            make_op("pointwise"), {map_ins.at(bytes), map_ins.at(inputs.back())}, {pm});
        for(auto&& [param, input] : oldm->get_ins_param_map(reduce->inputs(), true))
        {
            auto it        = std::find(reduce->inputs().begin(), reduce->inputs().end(), input);
            auto i         = std::distance(reduce->inputs().begin(), it);
            map_ins[param] = input == broadcast ? nibble : map_ins.at(inputs[i]);
        }
        auto outs = sm->add_instructions(oldm,
                                         &map_ins,
                                         reduce_reshape::transform_op(split_axis_op{
                                             axis, 2, broadcast->get_shape().ndim(), axes}));
        sm->add_return(outs);
        finalize_reduce_module(sm);
        return sm;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m        = mpm.get_module();
        auto reduce    = r.result;
        auto broadcast = r.instructions["broadcast"];
        auto unpack    = r.instructions["unpack"];
        auto axis      = find_split_axis(reduce, broadcast, unpack);
        if(not axis.has_value())
            return;
        auto [bytes, select] =
            insert_packed_inputs(m, reduce, broadcast, unpack->inputs().front(), *axis);
        // The split inputs in the reduce input order, the packed bytes in
        // place of the broadcast, then the nibble select
        std::vector<instruction_ref> inputs;
        auto tensors = reduce_tensor_inputs(reduce);
        for(auto input : reduce->inputs())
        {
            if(input == broadcast)
            {
                inputs.push_back(bytes);
                continue;
            }
            // Gather indices are not reduce inputs, so they are not split
            if(not contains(tensors, input))
            {
                inputs.push_back(input);
                continue;
            }
            auto split = insert_split_axis(m, reduce, input, *axis);
            if(not split.has_value())
                return;
            inputs.push_back(*split);
        }
        inputs.push_back(select);

        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        auto axes        = split_reduce_axes(reduce_axes, *axis);
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        auto* sm        = create_split_module(mpm, reduce, broadcast, inputs, *axis, axes, map_ins);
        auto new_inputs = find_inputs(map_ins, &m, sm);
        auto new_reduce = m.insert_instruction(reduce, fused_reduce{axes}, new_inputs, {sm});
        m.replace_instruction(reduce, make_op("squeeze", {{"axes", {*axis + 1}}}), new_reduce);
    }
};

} // namespace

void fuse_reduce::apply(module_pass_manager& mpm) const
{
    if(enabled(MIGRAPHX_DISABLE_REDUCE_FUSION{}))
        return;
    create_reduce_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    // Each fusion can expose the next one, so fuse until nothing changes
    const int max_iterations = 8;
    for(int i = 0; i < max_iterations; i++)
    {
        auto before = to_string(mpm.get_module());
        if(enable_rewrite_reshapes)
            mpm.run_pass(rewrite_reshapes<reduce_reshape>{});
        if(enable_rewrite_broadcasts)
        {
            match::find_matches(mpm, find_reduce_broadcast{});
            rewrite_broadcasts(mpm, "fused_reduce");
        }
        match::find_matches(mpm,
                            find_reduce_pointwise{},
                            find_pointwise_reduce{},
                            find_reduce_reduce{},
                            find_unpack_reduce{},
                            find_unpack_broadcast_reduce{},
                            find_reduce_slice{},
                            find_reduce_squeeze_pointwise{});
        mpm.run_pass(dead_code_elimination{});
        if(to_string(mpm.get_module()) == before)
            break;
    }
    // The gathers move in last: a reduce reading gathered inputs cant be
    // remapped by the reshape and broadcast rewrites the other fusions need
    if(enable_gather)
    {
        match::find_matches(mpm, find_gather_reduce{});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
