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
#include <migraphx/param_utils.hpp>
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
        auto output_shapes = sm->get_output_shapes();
        if(output_shapes.empty())
            MIGRAPHX_THROW("submodule has no outputs");
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

        // If all outputs are dynamic, return them directly
        if(std::all_of(output_shapes.begin(), output_shapes.end(), [](const shape& os) {
               return os.dynamic();
           }))
        {
            if(output_shapes.size() == 1)
                return output_shapes.front();
            return shape{output_shapes};
        }

        auto perm = find_permutation(inputs);
        std::vector<shape> result_shapes;
        std::transform(output_shapes.begin(),
                       output_shapes.end(),
                       std::back_inserter(result_shapes),
                       [&](const shape& os) {
                           if(os.dynamic())
                               return os;
                           return shape::from_permutation(os.type(), os.lens(), perm);
                       });
        if(result_shapes.size() == 1)
            return result_shapes.front();
        return shape{result_shapes};
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
               match::name("multibroadcast")(
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

static instruction_ref
merge_reduces(module_pass_manager& mpm, instruction_ref input, instruction_ref output)
{
    auto& m         = mpm.get_module();
    const auto* rm1 = input->module_inputs().front();
    const auto* rm2 = output->module_inputs().front();
    auto* rm        = mpm.create_module(rm1->name() + ":" + rm2->name());
    rm->set_bypass();

    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    auto outs1 = insert_module_in_submodule(rm, input, &map_ins);
    auto outs2 = insert_module_in_submodule(rm, output, &map_ins);

    std::vector<instruction_ref> all_outs;
    all_outs.insert(all_outs.end(), outs1.begin(), outs1.end());
    all_outs.insert(all_outs.end(), outs2.begin(), outs2.end());
    rm->replace_return(all_outs);
    finalize_reduce_module(rm);

    auto new_inputs = find_inputs(map_ins, &m, rm);
    auto fins       = m.insert_instruction(output, input->get_operator(), new_inputs, {rm});

    // Replace the first instruction's usages with get_tuple_elem
    if(input->get_shape().type() == shape::tuple_type)
    {
        auto input_outputs = input->outputs();
        for(auto inp_out : input_outputs)
        {
            if(inp_out->name() != "get_tuple_elem")
                continue;
            auto v = inp_out->get_operator().to_value();
            auto i = v.at("index").to<std::size_t>();
            m.replace_instruction(
                inp_out, make_op("get_tuple_elem", {{"index", i}}), fins);
        }
    }
    else
    {
        auto elem = m.insert_instruction(
            std::next(fins), make_op("get_tuple_elem", {{"index", 0}}), fins);
        m.replace_instruction(input, elem);
    }

    // Replace the second instruction's usages with get_tuple_elem
    std::size_t start2 = outs1.size();
    if(output->get_shape().type() == shape::tuple_type)
    {
        auto output_outputs = output->outputs();
        for(auto out_out : output_outputs)
        {
            if(out_out->name() != "get_tuple_elem")
                continue;
            auto v = out_out->get_operator().to_value();
            auto i = v.at("index").to<std::size_t>();
            m.replace_instruction(
                out_out, make_op("get_tuple_elem", {{"index", i + start2}}), fins);
        }
    }
    else
    {
        auto elem = m.insert_instruction(
            std::next(fins), make_op("get_tuple_elem", {{"index", start2}}), fins);
        m.replace_instruction(output, elem);
    }

    return fins;
}

static void try_multi_output_merge(module_pass_manager& mpm, instruction_ref reduce)
{
    if(reduce->outputs().empty())
        return;
    auto& m = mpm.get_module();

    // Collect sibling fused_reduces sharing an input with the same operator
    std::vector<instruction_ref> candidates;
    candidates.push_back(reduce);
    for(auto inp : reduce->inputs())
    {
        std::copy_if(
            inp->outputs().begin(),
            inp->outputs().end(),
            std::back_inserter(candidates),
            [&](instruction_ref output) {
                if(output == reduce)
                    return false;
                if(output->name() != "fused_reduce")
                    return false;
                if(not m.has_instruction(output))
                    return false;
                if(output->get_operator() != reduce->get_operator())
                    return false;
                if(output->outputs().empty())
                    return false;
                return std::find(candidates.begin(), candidates.end(), output) ==
                       candidates.end();
            });
    }

    if(candidates.size() < 2)
        return;

    // Sort by position in module
    std::sort(candidates.begin(), candidates.end(), by(std::less<>{}, [&](auto x) {
                  return std::distance(m.begin(), x);
              }));

    // Filter to independent instructions (no reachability between them)
    std::vector<instruction_ref> independent;
    std::copy_if(
        candidates.begin(), candidates.end(), std::back_inserter(independent), [&](auto c) {
            return std::none_of(independent.begin(), independent.end(), [&](auto other) {
                return reaches(other, c, &m);
            });
        });

    if(independent.size() < 2)
        return;

    // Iteratively merge all independent reduces
    (void)std::accumulate(
        independent.begin() + 1,
        independent.end(),
        independent.front(),
        [&](auto prev, auto next) { return merge_reduces(mpm, prev, next); });
}

namespace {
struct find_pointwise_reduce
{
    bool multi_output = false;

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
        if(multi_output)
            try_multi_output_merge(mpm, reduce);
    }
};

struct find_reduce_pointwise
{
    bool multi_output = false;

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
        if(multi_output)
            try_multi_output_merge(mpm, pw);
    }
};

struct find_reduce_reduce
{
    bool multi_output = false;

    auto matcher() const
    {
        return match::any_of(
            match::name("fused_reduce")(match_broadcastable_input("fused_reduce", "reduce")),
            match::name("fused_reduce"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce1 = r.result;

        if(contains(r.instructions, "reduce"))
        {
            // Re-validate broadcast axes since any_of may leave stale bindings
            if(contains(r.instructions, "broadcast"))
            {
                auto broadcast = r.instructions["broadcast"];
                auto axes =
                    reduce1->get_operator().to_value().at("axes").to_vector<std::size_t>();
                if(not is_valid_broadcast(broadcast, axes))
                {
                    if(multi_output)
                        try_multi_output_merge(mpm, reduce1);
                    return;
                }
            }

            // Chain fusion
            auto reduce2 = r.instructions["reduce"];
            auto input   = r.instructions["input"];

            if(reduce1->get_operator() != reduce2->get_operator())
            {
                if(multi_output)
                    try_multi_output_merge(mpm, reduce1);
                return;
            }

            const auto* rm1 = reduce1->module_inputs().front();
            const auto* rm2 = reduce2->module_inputs().front();
            auto* rm        = mpm.create_module(rm1->name() + ":" + rm2->name());
            rm->set_bypass();

            std::unordered_map<instruction_ref, instruction_ref> map_ins;
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
            mpm.get_module().replace_instruction(
                reduce1, reduce1->get_operator(), new_inputs, {rm});
        }

        if(multi_output)
            try_multi_output_merge(mpm, reduce1);
    }
};

struct reduce_reshape : rewrite_reshapes_base
{
    static std::string name() { return "fused_reduce"; }

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
            if(sop.name() == "multibroadcast")
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
        match::find_matches(mpm,
                            find_reduce_pointwise{.multi_output = enable_multi_output},
                            find_pointwise_reduce{.multi_output = enable_multi_output},
                            find_reduce_reduce{.multi_output = enable_multi_output});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
