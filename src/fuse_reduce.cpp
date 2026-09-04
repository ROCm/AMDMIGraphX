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
#include <migraphx/check_shapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/instruction.hpp>
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
#include <migraphx/functional.hpp>
#include <iterator>
#include <map>
#include <numeric>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_REDUCE_FUSION)

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
        if(not sm->bypass())
            MIGRAPHX_THROW("fused_reduce: bypass flag is not set");
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this, true}.has(names.size()).same_ndims();
        std::sort(names.begin(), names.end());
        auto shapes = sm->get_parameter_shapes();
        // Check dimension matches for each input
        if(not equal(names, inputs, [&](const auto& name, const auto& input) {
               auto s = shapes.at(name);
               return shape::same_lens(input, s);
           }))
            MIGRAPHX_THROW("Input dimension does not match the submodule.");

        if(sm->get_output_shapes().size() != 1)
        {
            auto result = sm->compute_shapes(
                inputs, {.name = name(), .strict_type = true, .strict_lens = true});
            return shape{result};
        }

        if(sm->get_output_shapes().front().dynamic())
            return sm->get_output_shapes().front();

        return shape::from_permutation(sm->get_output_shapes().front().type(),
                                       sm->get_output_shapes().front().lens(),
                                       find_permutation(inputs));
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
                if(not is_valid_broadcast(broadcast->get_shape(), axes))
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
        // Rebuilding the submodule at expanded shapes cant remap packed inputs
        if(has_unpack(reduce))
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
        auto reduce = r.result;
        auto input  = r.instructions["pointwise"];
        // Fusing the broadcast makes the reduction read purely broadcasted
        // data unless another input spans the reduce axes
        if(contains(r.instructions, "broadcast"))
        {
            auto axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
            if(not has_spanning_input(reduce->inputs(), r.instructions["final_broadcast"], axes))
                return;
        }
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

// Fusing a pointwise whose output is larger than what the reduction reads
// makes the kernel write that output with one workgroup per reduction
// output, so it needs at least min_fused_outputs outputs to stream it well;
// otherwise the pointwise is faster as a separate, fully parallel kernel
static bool
can_fuse_pointwise(instruction_ref reduce, instruction_ref pw, std::size_t min_fused_outputs)
{
    if(reduce->get_shape().elements() >= min_fused_outputs)
        return true;
    auto it = std::max_element(
        reduce->inputs().begin(), reduce->inputs().end(), by(std::less<>{}, [](instruction_ref i) {
            return i->get_shape().elements();
        }));
    return pw->get_shape().elements() <= (*it)->get_shape().elements();
}

struct find_reduce_pointwise
{
    std::size_t min_fused_outputs = 1;

    auto matcher() const
    {
        return match::name("pointwise")(match_broadcastable_input("fused_reduce", "reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto pw     = r.result;
        auto reduce = r.instructions["reduce"];
        auto input  = r.instructions["input"];
        if(not can_fuse_pointwise(reduce, pw, min_fused_outputs))
            return;

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
        auto axis = unpack->get_operator().to_value().at("axis").to<std::int64_t>();
        auto ndim = unpack->inputs().front()->get_shape().ndim();
        return axis < 0 ? axis + ndim : axis;
    }

    // Move the unpack below the reshapes between it and the reduce by
    // reshaping the packed input instead, so the unpack can be fused
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
        std::vector<std::int64_t> packed_dims(lens.begin(), lens.end());
        auto packed_reshape =
            m.insert_instruction(input, make_op("reshape", {{"dims", packed_dims}}), packed);
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
        if(not std::all_of(reduce->inputs().begin(), reduce->inputs().end(), [&](auto ri) {
               if(contains(fp8_types{}.get(), ri->get_shape().type()))
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
        // Submodules with packed inputs cant be remapped to the common dims
        return not has_unpack(ins);
    }

    template <class Transform>
    static auto transform_op(Transform t)
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
/// a gate_up matvec). Waits for the unpack to be fused since the slices are
/// pushed into the reduce inputs.
struct find_reduce_slice
{
    auto matcher() const
    {
        auto unit_reshapes = match::name("squeeze", "unsqueeze", "transpose");
        return match::name("slice")(
            match::arg(0)(match::skip(unit_reshapes)(match::name("fused_reduce").bind("reduce"))));
    }

    static std::vector<std::size_t> slice_axes(instruction_ref slice)
    {
        return slice->get_operator().to_value().at("axes").to_vector<std::size_t>();
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
        std::vector<operation> ops;
        for(auto ins = input; ins != reduce; ins = ins->inputs().front())
        {
            if(ins != input and ins->outputs().size() != 1)
                return;
            ops.push_back(ins->get_operator());
        }
        if(input != reduce and reduce->outputs().size() != 1)
            return;
        std::reverse(ops.begin(), ops.end());
        const auto& rlens = reduce->get_shape().lens();
        std::size_t raxis = axes.front();
        if(not ops.empty())
        {
            auto desc = shape_transform_descriptor::create(rlens, ops);
            if(desc.empty())
                return;
            auto is = range(rlens.size());
            auto it = std::find_if(is.begin(), is.end(), [&](auto i) {
                return desc.get_dst_axes_from_src(i) == std::vector<std::size_t>{axes.front()};
            });
            if(it == is.end())
                return;
            raxis = *it;
        }
        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        if(contains(reduce_axes, raxis))
            return;
        auto len = static_cast<std::int64_t>(rlens[raxis]);
        if(start < 0 or end <= start or end > len or end - start == len)
            return;
        if(not all_of(reduce->inputs(), [&](instruction_ref x) {
               return x->get_shape().lens()[raxis] == rlens[raxis];
           }))
            return;
        auto slice_op = make_op("slice", {{"axes", {raxis}}, {"starts", {start}}, {"ends", {end}}});
        auto inputs   = reduce->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](instruction_ref x) {
            return m.insert_instruction(slice, slice_op, x);
        });
        // Broadcasts inside the submodule expand to the full axis
        auto new_len     = static_cast<std::size_t>(end - start);
        const auto* oldm = reduce->module_inputs().front();
        auto* sm         = mpm.create_module(oldm->name() + "_slice" + std::to_string(start));
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
        auto y          = std::accumulate(
            ops.begin(), ops.end(), new_reduce, [&](instruction_ref ins, const operation& op) {
                return m.insert_instruction(slice, op, ins);
            });
        assert(y->get_shape().lens() == slice->get_shape().lens());
        m.replace_instruction(slice, y);
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
        if(pw->get_shape().type() == shape::tuple_type)
            return;
        auto is_squeezed_reduce = [&](instruction_ref input) {
            if(input->name() != "squeeze" or input->outputs().size() != 1)
                return false;
            if(input->inputs().front()->name() != "fused_reduce")
                return false;
            return squeeze_axes(input) == axes;
        };
        auto inputs = pw->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](instruction_ref input) {
            if(is_squeezed_reduce(input))
                return input->inputs().front();
            return m.insert_instruction(pw, make_op("unsqueeze", {{"axes", axes}}), input);
        });
        auto new_pw = m.insert_instruction(pw, pw->get_operator(), inputs, pw->module_inputs());
        m.replace_instruction(pw, make_op("squeeze", {{"axes", axes}}), new_pw);
    }
};

} // namespace

void fuse_reduce::apply(module_pass_manager& mpm) const
{
    if(enabled(MIGRAPHX_DISABLE_REDUCE_FUSION{}))
        return;
    create_reduce_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    for(int i = 0; i < 4; i++)
    {
        if(enable_rewrite_reshapes)
            mpm.run_pass(rewrite_reshapes<reduce_reshape>{});
        if(enable_rewrite_broadcasts)
        {
            match::find_matches(mpm, find_reduce_broadcast{});
            rewrite_broadcasts(mpm, "fused_reduce");
        }
        match::find_matches(mpm,
                            find_reduce_pointwise{.min_fused_outputs = min_fused_outputs},
                            find_pointwise_reduce{},
                            find_reduce_reduce{},
                            find_unpack_reduce{},
                            find_reduce_slice{},
                            find_reduce_squeeze_pointwise{});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
