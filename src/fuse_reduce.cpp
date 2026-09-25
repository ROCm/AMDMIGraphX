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
        if(sm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("Only one output supported");
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
        if(not std::all_of(reduce->inputs().begin(), reduce->inputs().end(), [&](auto ri) {
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

/// The lens with the axis split into (len / 2, 2); a unit axis stays unit
std::vector<std::size_t> split_axis_lens(std::vector<std::size_t> lens, std::size_t axis)
{
    assert(lens[axis] == 1 or lens[axis] % 2 == 0);
    std::size_t n = lens[axis] == 1 ? 1 : 2;
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

/// A view of the input with the axis split in two: a broadcast is rebuilt
/// from its input so the reshape stays a view
optional<instruction_ref>
insert_split_axis(module& m, instruction_ref pos, instruction_ref input, std::size_t axis)
{
    const auto& s = input->get_shape();
    if(s.standard())
        return m.insert_instruction(
            pos, make_op("reshape", {{"dims", split_axis_lens(s.lens(), axis)}}), input);
    if(not contains({"multibroadcast", "broadcast"}, input->name()))
        return nullopt;
    auto bin = input->inputs().front();
    if(not bin->get_shape().standard())
        return nullopt;
    auto lens = broadcast_input_lens(s);
    if(elements(lens) != bin->get_shape().elements())
        return nullopt;
    auto r =
        m.insert_instruction(pos, make_op("reshape", {{"dims", split_axis_lens(lens, axis)}}), bin);
    return m.insert_instruction(
        pos, make_op("multibroadcast", {{"out_lens", split_axis_lens(s.lens(), axis)}}), r);
}

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

    static std::size_t op_axis(const operation& op, std::size_t ndim)
    {
        return tune_axis(ndim, op.to_value().at("axis").to<int>(), op.name());
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
        if(not all_of(reduce->inputs(), [&](instruction_ref input) {
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
        auto remap_axis = [&](std::size_t a) -> std::int64_t { return a > axis ? a + 1 : a; };
        auto outs       = sm->add_instructions(
            oldm, &map_ins, reduce_reshape::transform_op([&](const operation& sop) {
                auto v = sop.to_value();
                if(contains(sop.name(), "reduce"))
                    return make_op(sop.name(), {{"axes", axes}});
                if(contains({"argmin", "argmax", "unpack_int4"}, sop.name()))
                {
                    v["axis"] = remap_axis(op_axis(sop, broadcast->get_shape().ndim()));
                    return make_op(sop.name(), v);
                }
                if(contains({"multibroadcast", "broadcast"}, sop.name()))
                {
                    auto out_lens = v.at("out_lens").to_vector<std::size_t>();
                    return make_op("multibroadcast",
                                   {{"out_lens", split_axis_lens(out_lens, axis)}});
                }
                return sop;
            }));
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
        for(auto input : reduce->inputs())
        {
            if(input == broadcast)
            {
                inputs.push_back(bytes);
                continue;
            }
            auto split = insert_split_axis(m, reduce, input, *axis);
            if(not split.has_value())
                return;
            inputs.push_back(*split);
        }
        inputs.push_back(select);

        auto reduce_axes = reduce->get_operator().to_value().at("axes").to_vector<std::size_t>();
        std::vector<std::int64_t> axes;
        for(auto a : reduce_axes)
        {
            axes.push_back(a > *axis ? a + 1 : a);
            if(a == *axis)
                axes.push_back(a + 1);
        }
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
                            find_reduce_pointwise{},
                            find_pointwise_reduce{},
                            find_reduce_reduce{},
                            find_unpack_reduce{},
                            find_unpack_broadcast_reduce{});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
