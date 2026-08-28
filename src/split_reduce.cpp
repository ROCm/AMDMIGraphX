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
#include <migraphx/split_reduce.hpp>
#include <migraphx/dom_info.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/split_factor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct split_fused_reduce
{
    std::vector<std::int64_t> axes{};
    std::string assign = "assign_none";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.assign, "assign"));
    }

    value attributes() const { return {{"prefill", 0}}; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        const auto* sm = mods.front();
        auto names     = sm->get_parameter_names();
        check_shapes{inputs, *this}.has(names.size()).same_ndims();

        auto result =
            sm->compute_shapes(inputs, {.name = name(), .strict_type = true, .strict_lens = true});
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }

    std::string name() const { return "split_fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(split_fused_reduce);

static bool is_reduce(const instruction& ins)
{
    return contains(ins.name(), "reduce") or ins.name() == "argmin" or ins.name() == "argmax";
}

namespace {
struct splitter
{
    const_module_ref rm;

    bool strictly_dominate(instruction_ref a, instruction_ref b)
    {
        if(not dom.has_value())
            dom = compute_dominator(*rm);
        return dom->strictly_dominate(a, b);
    }

    std::vector<instruction_ref> find_splits() const
    {
        std::vector<instruction_ref> result;
        copy_if(iterator_for(*rm), std::back_inserter(result), [](auto ins) {
            return is_reduce(*ins);
        });
        if(result.size() > 2)
            return {};
        // argmin/argmax cant be completed by a second reduction
        if(not std::all_of(result.begin(), result.end(), [](instruction_ref ins) {
               return contains(ins->name(), "reduce");
           }))
            return {};
        if(result.size() < 2)
            return result;
        // Skip internal reductions(like softmax's reduce_max) since the
        // partial result cant be completed by the trailing reduction
        if(reaches(result[0], result[1]))
            return {};
        return result;
    }

    std::vector<instruction_ref> find_alive(const std::vector<instruction_ref>& splits)
    {
        std::vector<instruction_ref> result;
        bool stop = false;
        liveness(*rm, [&](auto rins, const auto& live_set) {
            if(stop)
                return;
            if(rins == rm->begin())
                return;
            // We want to know what instructions are live after the split instruction
            auto aliases = instruction::get_output_alias(std::prev(rins));
            if(not std::any_of(aliases.begin(), aliases.end(), [&](instruction_ref ins) {
                   return contains(splits, ins);
               }))
                return;
            std::copy_if(live_set.begin(),
                         live_set.end(),
                         std::back_inserter(result),
                         [&](instruction_ref live) {
                             if(live->name() == "@param")
                                 return false;
                             if(contains(splits, live))
                                 return false;
                             if(splits.size() > 1 and none_of(splits, [&](instruction_ref split) {
                                    return this->strictly_dominate(live, split);
                                }))
                                 return false;
                             return true;
                         });
            stop = true;
        });
        return result;
    }

    std::optional<dominator_info> dom = std::nullopt;
};

struct partial_split
{
    std::int64_t axis = 0;
    std::size_t group = 1;

    // Split dimension `axis` into {group, n/group} so the first fused_reduce
    // can do a partial reduction over the remaining n/group elements, which
    // stay contiguous so the reads coalesce. Broadcast dimensions stay 1.
    std::vector<std::size_t> split_dims(std::vector<std::size_t> dims) const
    {
        auto& dim      = dims[axis];
        auto group_dim = dim == 1 ? 1 : group;
        dim /= group_dim;
        dims.insert(dims.begin() + axis, group_dim);
        return dims;
    }

    // Shift the axes to account for the inserted group dimension, which is
    // not reduced by the first fused_reduce
    std::vector<std::int64_t> split_axes(std::vector<std::int64_t> axes) const
    {
        std::transform(axes.begin(), axes.end(), axes.begin(), [&](std::int64_t a) {
            return a < axis ? a : a + 1;
        });
        return axes;
    }

    operation split_op(const operation& op) const
    {
        auto v = op.to_value();
        if(contains(op.name(), "reduce"))
        {
            v["axes"] = split_axes(v["axes"].to_vector<std::int64_t>());
            return make_op(op.name(), v);
        }
        if(op.name() == "multibroadcast")
        {
            v["out_lens"] = split_dims(v["out_lens"].to_vector<std::size_t>());
            return make_op(op.name(), v);
        }
        return op;
    }
};
} // namespace

static std::string assign_op(const std::vector<instruction_ref>& splits)
{
    static std::unordered_map<std::string, std::string> m = {
        {"reduce_sum", "assign_add"},
        {"reduce_mean", "assign_add"},
        {"reduce_prod", "assign_mul"},
        {"reduce_max", "assign_max"},
        {"reduce_min", "assign_min"},
    };
    return m.at(splits.front()->name());
}

static std::vector<instruction_ref>
insert_module_inline(module& m, instruction_ref ins, const module::with_inputs& mwi)
{
    auto param_map = mwi.mod.get_ins_param_map(mwi.inputs, true);
    return m.insert_instructions(ins, &mwi.mod, &param_map);
}

// Get each output of a multi-output instruction, which returns a tuple when
// there is more than one output
static std::vector<instruction_ref>
insert_tuple_elements(module& m, instruction_ref ins, instruction_ref value, std::size_t n)
{
    if(n == 1)
        return {value};
    std::vector<instruction_ref> result;
    transform(range(n), std::back_inserter(result), [&](auto i) {
        return m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", i}}), value);
    });
    return result;
}

static std::size_t get_reduce_size(const_module_ref rm)
{
    auto ins = std::find_if(rm->begin(), rm->end(), &is_reduce);
    assert(ins != rm->end());
    return ins->inputs().front()->get_shape().elements() / ins->get_shape().elements();
}

// Atomic-based split_fused_reduce only supports float reduce_sum for now
// TODO: Support other reduction types and data types
static bool can_use_atomic_split(const std::vector<instruction_ref>& splits)
{
    return std::all_of(splits.begin(), splits.end(), [](instruction_ref split) {
        return split->name() == "reduce_sum" and
               contains({shape::float_type, shape::half_type}, split->get_shape().type());
    });
}

static std::optional<partial_split> find_partial_split(const_module_ref rm,
                                                       const std::vector<instruction_ref>& splits,
                                                       const std::vector<std::int64_t>& axes,
                                                       std::size_t lower_split_size)
{
    // Every operator must be mappable onto the split dimensions
    if(not std::all_of(rm->begin(), rm->end(), [](const instruction& i) {
           return is_reduce(i) or
                  contains({"@param", "@return", "pointwise", "multibroadcast"}, i.name());
       }))
        return std::nullopt;
    // The trailing reduction completes each partial result with the same axes
    if(not std::all_of(splits.begin(), splits.end(), [&](instruction_ref split) {
           return split->get_operator().to_value()["axes"].to_vector<std::int64_t>() == axes;
       }))
        return std::nullopt;
    auto it   = std::max_element(splits.begin(), splits.end(), by(std::less<>{}, [](auto split) {
                                   return split->inputs().front()->get_shape().elements();
                                 }));
    auto lens = (*it)->inputs().front()->get_shape().lens();
    auto relements = transform_accumulate(
        axes.begin(), axes.end(), std::size_t{1}, std::multiplies<>{}, [&](auto axis) {
            return lens[axis];
        });
    // Pick the reduce axis that can be split into the most groups, preferring
    // the innermost axis on ties. The threshold is scaled by the reduction
    // size of the other axes so the remaining reduction is below the
    // lower_split_size.
    auto best = transform_accumulate(
        axes.begin(),
        axes.end(),
        partial_split{},
        [](const partial_split& x, const partial_split& y) { return y.group >= x.group ? y : x; },
        [&](std::int64_t axis) -> partial_split {
            std::size_t r = lens[axis];
            auto min_size = std::max<std::size_t>(
                lower_split_size / std::max<std::size_t>(relements / r, 1), 1);
            return {axis, split_dim(r, min_size)};
        });
    if(best.group < 2)
        return std::nullopt;
    // Every shape must be broadcast or full along the split axis
    if(not std::all_of(rm->begin(), rm->end(), [&](const instruction& i) {
           if(i.name() == "@return")
               return true;
           auto dim = i.get_shape().lens()[best.axis];
           return dim == 1 or dim == lens[best.axis];
       }))
        return std::nullopt;
    return best;
}

static void apply_partial_split(module_pass_manager& mpm,
                                instruction_ref ins,
                                const std::vector<instruction_ref>& splits,
                                const std::vector<std::int64_t>& axes,
                                const partial_split& ps,
                                std::array<module::with_inputs, 2> mods)
{
    auto& m  = mpm.get_module();
    auto* rm = ins->module_inputs().front();

    // Reshape the inputs so the split axis becomes {n/group, group}
    std::vector<instruction_ref> split_inputs;
    std::transform(mods[0].inputs.begin(),
                   mods[0].inputs.end(),
                   std::back_inserter(split_inputs),
                   [&](instruction_ref input) {
                       return m.insert_instruction(
                           ins,
                           make_op("reshape", {{"dims", ps.split_dims(input->get_shape().lens())}}),
                           input);
                   });

    // The first fused_reduce does a partial reduction for each group
    auto* splitm = mpm.create_module(rm->name() + "_split");
    splitm->set_bypass();
    auto outs =
        splitm->fuse(mods[0].mod,
                     split_inputs,
                     nullptr,
                     [&](module& sm,
                         instruction_ref pos,
                         const operation& op,
                         const std::vector<instruction_ref>& inputs,
                         const std::vector<module_ref>& mod_args) {
                         return sm.insert_instruction(pos, ps.split_op(op), inputs, mod_args);
                     });
    splitm->add_return(outs);
    auto partial = m.insert_instruction(
        ins, make_op("fused_reduce", {{"axes", ps.split_axes(axes)}}), split_inputs, {splitm});
    auto partials = insert_tuple_elements(m, ins, partial, splits.size());

    // Squeeze the reduced axis so the group dimension takes its place
    std::vector<instruction_ref> squeezed;
    std::transform(
        partials.begin(), partials.end(), std::back_inserter(squeezed), [&](instruction_ref p) {
            return m.insert_instruction(ins, make_op("squeeze", {{"axes", {ps.axis + 1}}}), p);
        });

    // The second fused_reduce completes each partial result with another
    // reduction over the groups
    auto* finalm = mpm.create_module(rm->name() + "_final");
    finalm->set_bypass();
    std::vector<instruction_ref> completed;
    transform(range(squeezed.size()), std::back_inserter(completed), [&](auto i) {
        auto param = finalm->add_parameter(param_name(i), squeezed[i]->get_shape().as_standard());
        return finalm->add_instruction(make_op(splits[i]->name(), {{"axes", axes}}), param);
    });

    // The completion kernel runs one workgroup per reduction output, which
    // is enough to stream a full-sized output when there are at least this
    // many outputs
    const std::size_t min_fused_outputs = 8;
    auto noutputs                       = splits.front()->get_shape().elements();
    if(ins->get_shape().elements() <= noutputs or noutputs >= min_fused_outputs)
    {
        // Fuse the trailing operators into the completion kernel to avoid
        // another kernel launch. The partials have a different shape than
        // the splits they replace, so with_inputs::replace cant be used.
        std::transform(mods[1].inputs.begin(),
                       mods[1].inputs.end(),
                       mods[1].inputs.begin(),
                       [&](instruction_ref input) {
                           auto it = std::find(splits.begin(), splits.end(), input);
                           if(it == splits.end())
                               return input;
                           return squeezed[it - splits.begin()];
                       });
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        std::transform(squeezed.begin(),
                       squeezed.end(),
                       completed.begin(),
                       std::inserter(map_ins, map_ins.end()),
                       [](instruction_ref sq, instruction_ref c) { return std::make_pair(sq, c); });
        finalm->add_return(finalm->fuse(mods[1].mod, mods[1].inputs, &map_ins));
        auto replaced = m.insert_instruction(
            ins, make_op("fused_reduce", {{"axes", axes}}), mods[1].inputs, {finalm});
        m.replace_instruction(ins, replaced);
    }
    else
    {
        // With so few workgroups the completion kernel would starve the
        // device writing the full-sized output. Complete the reduction
        // alone and insert the trailing operators into the parent module so
        // they can run as a fully parallel pointwise kernel.
        finalm->add_return(completed);
        auto completion = m.insert_instruction(
            ins, make_op("fused_reduce", {{"axes", axes}}), squeezed, {finalm});
        mods[1].replace(splits, insert_tuple_elements(m, ins, completion, splits.size()));
        auto replaced = insert_module_inline(m, ins, mods[1]);
        assert(replaced.size() == 1);
        m.replace_instruction(ins, replaced.front());
    }
}

static void apply_atomic_split(module_pass_manager& mpm,
                               instruction_ref ins,
                               const std::vector<instruction_ref>& splits,
                               const std::vector<std::int64_t>& axes,
                               std::array<module::with_inputs, 2> mods)
{
    auto* rm     = ins->module_inputs().front();
    auto* splitm = mpm.create_module(rm->name() + "_split", std::move(mods[0].mod));
    splitm->set_bypass();

    // Insert split reduce
    auto split_reduce = mpm.get_module().insert_instruction(
        ins,
        make_op("split_fused_reduce", {{"axes", axes}, {"assign", assign_op(splits)}}),
        mods[0].inputs,
        {splitm});

    mods[1].replace(splits,
                    insert_tuple_elements(mpm.get_module(), ins, split_reduce, splits.size()));
    auto replaced = insert_module_inline(mpm.get_module(), ins, mods[1]);
    assert(replaced.size() == 1);
    mpm.get_module().replace_instruction(ins, replaced.front());
}

void split_reduce::apply(module_pass_manager& mpm) const
{
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins->name() != "fused_reduce")
            continue;
        if(ins->get_shape().dynamic())
            continue;
        auto* rm         = ins->module_inputs().front();
        auto reduce_size = get_reduce_size(rm);
        if(reduce_size < split_size and reduce_size < lower_split_size)
            continue;
        splitter s{rm};
        auto splits = s.find_splits();
        if(splits.empty())
            continue;
        auto v    = ins->get_operator().to_value();
        auto axes = v["axes"].to_vector<std::int64_t>();

        auto batch = splits.front()->get_shape().elements();
        std::optional<partial_split> ps;
        // Below the upper_split_size a single workgroup per output can
        // handle the reduction well, so only split when the batch is too
        // small to fill the device. Beyond the upper_split_size the
        // reduction is too large for a single workgroup(the register limits
        // force the block_large fallback), so a split is needed regardless
        // of the batch.
        if(reduce_size >= lower_split_size and
           (batch < partial_max_batch or reduce_size >= upper_split_size))
            ps = find_partial_split(rm, splits, axes, lower_split_size);
        bool use_atomic = reduce_size >= split_size and can_use_atomic_split(splits);
        // When both thresholds are applicable, prefer_partial_reduce decides
        if(ps.has_value() and use_atomic and not prefer_partial_reduce)
            ps = std::nullopt;
        if(not ps.has_value() and not use_atomic)
            continue;

        auto alive = s.find_alive(splits);

        std::array<module::with_inputs, 2> mods;
        if(not alive.empty())
        {
            auto mods3 = rm->split(ins->inputs(), alive, splits);
            auto r     = insert_module_inline(mpm.get_module(), ins, mods3[0]);
            mods3[1].replace(alive, r);
            mods3[2].replace(alive, r);
            mods = {std::move(mods3[1]), std::move(mods3[2])};
        }
        else
        {
            mods = rm->split(ins->inputs(), splits);
        }

        if(ps.has_value())
            apply_partial_split(mpm, ins, splits, axes, *ps, std::move(mods));
        else
            apply_atomic_split(mpm, ins, splits, axes, std::move(mods));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
