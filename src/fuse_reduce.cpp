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
#include <migraphx/algorithm.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/stringutils.hpp>
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

// Reduce axes of an operation, handling argmin/argmax which use "axis"
static std::vector<std::int64_t> get_reduce_axes(const operation& op)
{
    auto v = op.to_value();
    std::vector<std::int64_t> axes;
    if(v.contains("axes"))
    {
        axes = v["axes"].to_vector<std::int64_t>();
    }
    else if(v.contains("axis"))
    {
        axes = {v["axis"].to<std::int64_t>()};
    }
    std::sort(axes.begin(), axes.end());
    return axes;
}

static bool is_reduce_ins(const instruction& ins)
{
    return ins.get_operator().attributes().get("reduce", false);
}

// A fused_reduce module contains a sub-reduction when its reduces use
// different axes
static bool has_sub_reduce(const_module_ref rm)
{
    std::vector<std::vector<std::int64_t>> axes;
    transform_if(
        rm->begin(),
        rm->end(),
        std::back_inserter(axes),
        [](const auto& ins) { return is_reduce_ins(ins); },
        [](const auto& ins) { return get_reduce_axes(ins.get_operator()); });
    return std::adjacent_find(axes.begin(), axes.end(), std::not_equal_to<>{}) != axes.end();
}

static void create_reduce_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not is_reduce_ins(*ins))
            continue;
        if(ins->inputs().size() != 1)
            continue;

        auto* rm =
            mpm.create_module(mpm.get_module().name() + ":" + ins->name() + std::to_string(n++));
        rm->set_bypass();

        rm->add_return(rm->fuse({ins}));

        mpm.get_module().replace_instruction(
            ins,
            make_op("fused_reduce", {{"axes", get_reduce_axes(ins->get_operator())}}),
            ins->inputs(),
            {rm});
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

namespace {
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

        // A broadcasted input would insert a stage-1 broadcast into a nested
        // reduce module, which the kernel generator cant re-slice
        if(contains(r.instructions, "broadcast") and has_sub_reduce(old_rm))
            return;

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

// Stage-1 reduce ops that the reduce kernel generator can emit as a
// sub-reduction: their write stage is an identity so it passes through the
// lazy storage unchanged
bool is_sub_reducible(const instruction& ins)
{
    if(contains({"reduce_sum", "reduce_max", "reduce_min", "reduce_prod"}, ins.name()))
        return true;
    if(ins.name() == "reduce_mean")
        return contains({shape::float_type, shape::half_type, shape::double_type},
                        ins.get_shape().type());
    return false;
}

/// Check if reduce2 can be fused into reduce1 as a nested sub-reduction.
/// reduce1 must consume only reduce2, reduce2's axes must be a subset of
/// reduce1's axes, and the modules must be limited to what the backend can
/// generate for a sub-reduction: no stage-1 broadcasts, a single sub-reduce
/// shape, and stage-2 broadcasts only of the final reduced value.
bool is_fusable_sub_reduce(instruction_ref reduce1, instruction_ref reduce2, std::size_t split_size)
{
    if(reduce1->inputs().size() != 1)
        return false;
    // Broadcasted inputs would be read as per-output scalars by the kernel
    // generator, which is wrong for the intermediate stage of a nested reduce
    if(std::any_of(reduce2->inputs().begin(), reduce2->inputs().end(), [](instruction_ref input) {
           return input->get_shape().dynamic() or input->get_shape().broadcasted();
       }))
        return false;
    auto axes1 = get_reduce_axes(reduce1->get_operator());
    auto axes2 = get_reduce_axes(reduce2->get_operator());
    if(not std::includes(axes1.begin(), axes1.end(), axes2.begin(), axes2.end()))
        return false;
    // Fusing collapses the intermediate parallelism into one workgroup per
    // final output, so limit it to small reductions
    auto input =
        std::max_element(reduce2->inputs().begin(),
                         reduce2->inputs().end(),
                         by(std::less<>{}, [](auto i) { return i->get_shape().elements(); }));
    const auto& lens = (*input)->get_shape().lens();
    auto relements   = transform_accumulate(
        axes1.begin(), axes1.end(), std::size_t{1}, std::multiplies<>{}, [&](auto axis) {
            return lens[axis];
        });
    if(relements > split_size)
        return false;
    const auto* rm1 = reduce1->module_inputs().front();
    const auto* rm2 = reduce2->module_inputs().front();
    // Stage-1 broadcasts would need re-slicing in the kernel
    if(std::any_of(rm2->begin(), rm2->end(), [&](const auto& ins) {
           if(contains(ins.name(), "broadcast"))
               return true;
           if(not is_reduce_ins(ins))
               return false;
           return not is_sub_reducible(ins) or get_reduce_axes(ins.get_operator()) != axes2;
       }))
        return false;
    // All sub-reduces must share the same input dimensions so the kernel can
    // use a single sub-reduce shape
    std::vector<std::vector<std::size_t>> sub_lens;
    transform_if(
        rm2->begin(),
        rm2->end(),
        std::back_inserter(sub_lens),
        [](const auto& ins) { return is_reduce_ins(ins); },
        [](const auto& ins) { return ins.inputs().front()->get_shape().lens(); });
    if(std::adjacent_find(sub_lens.begin(), sub_lens.end(), std::not_equal_to<>{}) !=
       sub_lens.end())
        return false;
    const auto& out_lens = rm1->get_output_shapes().front().lens();
    return std::none_of(rm1->begin(), rm1->end(), [&](const auto& ins) {
        if(contains(ins.name(), "broadcast"))
            return ins.inputs().front()->get_shape().lens() != out_lens;
        if(not is_reduce_ins(ins))
            return false;
        return not starts_with(ins.name(), "reduce_") or
               get_reduce_axes(ins.get_operator()) != axes1;
    });
}

struct find_reduce_reduce
{
    std::size_t split_size = 8192;

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
        {
            // Sequential reduces over different axes can be fused as a nested
            // sub-reduction when reduce1 directly consumes reduce2
            if(contains(r.instructions, "broadcast"))
                return;
            if(not is_fusable_sub_reduce(reduce1, reduce2, split_size))
                return;
        }

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

struct reduce_reshape : rewrite_reshapes_base
{
    static std::string name() { return "fused_reduce"; }

    template <class AxesMap>
    static std::vector<int64_t> remap_axes(const std::vector<int64_t>& old_axes, const AxesMap& am)
    {
        std::vector<int64_t> axes;
        for(auto axis : old_axes)
        {
            const auto& new_axes = am.at(axis);
            axes.insert(axes.end(), new_axes.begin(), new_axes.end());
        }
        std::sort(axes.begin(), axes.end());
        return axes;
    }

    // Remap broadcast out_lens axis-by-axis: broadcasted axes take the base
    // dims while axes of extent 1 stay 1, so a nested module's stage-2
    // broadcasts are not inflated to the full reduction size
    template <class AxesMap>
    static std::vector<std::size_t> remap_broadcast_lens(const std::vector<std::size_t>& old_lens,
                                                         const AxesMap& am,
                                                         const std::vector<std::size_t>& dims)
    {
        std::vector<std::size_t> lens(dims.size(), 1);
        for(auto axis : range(old_lens.size()))
        {
            if(old_lens[axis] == 1)
                continue;
            for(auto new_axis : am.at(axis))
                lens[new_axis] = dims[new_axis];
        }
        return lens;
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
        auto op    = any_cast<fused_reduce>(ins->get_operator());
        auto axes  = remap_axes(op.axes, am);
        auto dims  = base_dims(inputs);
        auto* oldm = ins->module_inputs().front();
        auto* sm   = mpm.create_module(oldm->name() + "_reshape");
        sm->set_bypass();
        auto outs = sm->fuse(*oldm, inputs, nullptr, transform_op([&](const operation& sop) {
            // remap each reduce by its own axes since a nested module can
            // contain sub-reduces over a subset of the axes
            if(contains(sop.name(), "reduce"))
            {
                auto v    = sop.to_value();
                v["axes"] = remap_axes(v["axes"].to_vector<std::int64_t>(), am);
                return make_op(sop.name(), v);
            }
            // handle argmin/argmax
            if(sop.name() == "argmin" or sop.name() == "argmax")
            {
                auto v    = sop.to_value();
                v["axis"] = axes.front();
                return make_op(sop.name(), v);
            }
            if(sop.name() == "multibroadcast")
            {
                auto v = sop.to_value();
                return make_op(
                    "multibroadcast",
                    {{"out_lens",
                      remap_broadcast_lens(v["out_lens"].to_vector<std::size_t>(), am, dims)}});
            }
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
        match::find_matches(
            mpm, find_reduce_pointwise{}, find_pointwise_reduce{}, find_reduce_reduce{split_size});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
