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
#include <migraphx/gpu/fuse_concat_past_present.hpp>
#include <migraphx/gpu/precompile_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// A size-1 slice of the input along axis at a runtime index. An index outside
// the axis is invalid input: concat_past_present skips such writes, but a view
// cannot skip, so it throws instead of silently targeting another slot.
struct slice_at
{
    std::size_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "gpu::slice_at"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& s = inputs.front();
        if(axis >= s.ndim())
            MIGRAPHX_THROW("SLICE_AT: axis out of range");
        if(inputs.back().elements() != 1 or not shape::is_integral(inputs.back().type()))
            MIGRAPHX_THROW("SLICE_AT: index must be a single integer");
        auto lens  = s.lens();
        lens[axis] = 1;
        return {s.type(), lens, s.strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto& s = args[0].get_shape();
        auto idx      = args[1].at<std::int64_t>();
        if(idx < 0 or idx >= static_cast<std::int64_t>(s.lens()[axis]))
            MIGRAPHX_THROW("SLICE_AT: index " + std::to_string(idx) + " out of range [0, " +
                           std::to_string(s.lens()[axis]) + ")");
        std::vector<std::size_t> start(s.ndim(), 0);
        start[axis]       = idx;
        auto offset       = s.index(start) * s.type_size();
        const auto& input = args[0];
        return {output_shape, [=] { return input.data() + offset; }};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};
MIGRAPHX_REGISTER_OP(slice_at);

namespace {

// concat_past_present fixes the (batch, kv_heads, seq, head) layout; seq is the append axis
constexpr std::size_t seq_axis = 2;

bool is_pointwise(instruction_ref ins)
{
    return ins->get_operator().to_value().at("op").at("name").to<std::string>() == "pointwise";
}

struct find_concat_past_present
{
    auto matcher() const
    {
        auto view =
            match::name("unsqueeze", "squeeze", "transpose", "reshape_lazy")(match::used_once());
        auto producer =
            precompile_name("pointwise", "fused_concat")(match::used_once()).bind("producer");
        return precompile_name("concat_past_present")(match::nargs(3),
                                                      match::arg(0)(match::skip(view)(producer)));
    }

    // The cache slot the append writes, in the layout of cur
    static shape slot_shape(instruction_ref cur, instruction_ref cache)
    {
        const auto& ks = cache->get_shape();
        auto lens      = ks.lens();
        lens[seq_axis] = cur->get_shape().lens()[seq_axis];
        return {ks.type(), lens, ks.strides()};
    }

    // The concat input must cover exactly one cache slot
    static bool fills_slot(instruction_ref cur, instruction_ref cache)
    {
        const auto& cs = cur->get_shape();
        const auto& ks = cache->get_shape();
        if(not ks.standard() or cs.type() != ks.type())
            return false;
        if(cs.ndim() != 4 or ks.ndim() != 4)
            return false;
        return cs.lens() == slot_shape(cur, cache).lens() and
               cs.lens()[seq_axis] <= ks.lens()[seq_axis];
    }

    // The shape after the view chain, if it applies without copying
    static optional<shape> view_shape(const shape& s, const std::vector<operation>& ops)
    {
        try
        {
            return std::accumulate(
                ops.begin(), ops.end(), s, [](const shape& acc, const operation& op) {
                    return op.compute_shape({acc});
                });
        }
        catch(const migraphx::exception&)
        {
            return nullopt;
        }
    }

    // One host copy of the sequence length serves every concat that reads it
    static instruction_ref load_scalar(module& m, instruction_ref ins, instruction_ref slk)
    {
        auto it = std::find_if(slk->outputs().begin(), slk->outputs().end(), [](auto out) {
            return out->name() == "hip::load_scalar";
        });
        if(it != slk->outputs().end())
            return *it;
        return m.insert_instruction(ins, make_op("hip::load_scalar"), slk);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto producer = r.instructions["producer"];
        auto cur      = ins->inputs()[0];
        auto slk      = ins->inputs()[1];
        auto cache    = ins->inputs()[2];
        if(not fills_slot(cur, cache))
            return;
        assert(not producer->inputs().empty());
        auto alloc = producer->inputs().back();
        if(alloc->name() != "allocate" or alloc->outputs().size() != 1)
            return;

        // The producer writes in its own dims and layout, so the slot is
        // re-indexed back through the views between it and the concat
        auto path = get_input_path(cur);
        std::vector<instruction_ref> chain(path.begin(),
                                           std::find(path.begin(), path.end(), producer));
        std::vector<operation> chain_ops;
        std::transform(chain.rbegin(),
                       chain.rend(),
                       std::back_inserter(chain_ops),
                       [](instruction_ref x) { return x->get_operator(); });
        auto inverse = invert_view_chain(chain_ops, producer->get_shape(), slot_shape(cur, cache));
        if(not inverse.has_value())
            return;

        auto seq = cur->get_shape().lens()[seq_axis];
        instruction_ref view;
        if(seq == 1)
        {
            // Decode appends at a device-computed position: read it on the
            // host and slice the cache at that offset
            if(cur->get_shape().lens()[0] != 1 or slk->get_shape().elements() != 1)
                return;
            view = m.insert_instruction(ins,
                                        make_op("gpu::slice_at", {{"axis", seq_axis}}),
                                        cache,
                                        load_scalar(m, ins, slk));
        }
        else
        {
            // Prompt appends at position zero, which is a static view
            view = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {seq_axis}}, {"starts", {0}}, {"ends", {seq}}}),
                cache);
        }

        // A retargeted copy of the producer is inserted at the concat rather
        // than rewriting it in place: the new strides could make the original
        // view chain fail shape recomputation before it becomes dead
        assert(view->get_shape() == slot_shape(cur, cache));
        view = std::accumulate(inverse->first.begin(),
                               inverse->first.end(),
                               view,
                               [&](instruction_ref start, const operation& op) {
                                   return m.insert_instruction(ins, op, start);
                               });
        assert(view->get_shape() == inverse->second);
        auto v            = producer->get_operator().to_value();
        v["output_shape"] = to_value(inverse->second);
        auto inputs       = producer->inputs();
        inputs.back()     = view;
        auto fused        = m.insert_instruction(
            ins, make_op("gpu::precompile_op", v), inputs, producer->module_inputs());
        m.replace_instruction(ins, make_op("identity"), {cache, fused});
    }
};

shape to_scalar(const shape& s) { return shape{s.type()}; }

// The buffers an instruction's result lives in
std::vector<instruction_ref> buffers(instruction_ref ins)
{
    return instruction::get_output_alias(ins);
}

bool shares_buffer(instruction_ref ins, const std::vector<instruction_ref>& bufs)
{
    auto xs = buffers(ins);
    return std::any_of(xs.begin(), xs.end(), [&](instruction_ref x) { return contains(bufs, x); });
}

bool reads_buffer(instruction_ref ins, const std::vector<instruction_ref>& bufs)
{
    return std::any_of(ins->inputs().begin(), ins->inputs().end(), [&](instruction_ref x) {
        return shares_buffer(x, bufs);
    });
}

// An append already retargeted to write its cache slot, found by the identity
// that orders the readers of the cache after the write
struct fused_append
{
    instruction_ref dep;
    instruction_ref producer;
    instruction_ref cache;

    static optional<fused_append> from_dep(instruction_ref ins)
    {
        if(ins->name() != "identity" or ins->inputs().size() != 2)
            return nullopt;
        auto cache    = ins->inputs()[0];
        auto producer = ins->inputs()[1];
        if(producer->name() != "gpu::precompile_op" or not is_pointwise(producer))
            return nullopt;
        if(producer->outputs().size() != 1 or producer->get_shape().type() == shape::tuple_type)
            return nullopt;
        if(producer->inputs().size() < 2 or
           not shares_buffer(producer->inputs().back(), buffers(cache)))
            return nullopt;
        return fused_append{ins, producer, cache};
    }

    std::vector<instruction_ref> tensor_inputs() const
    {
        return {producer->inputs().begin(), std::prev(producer->inputs().end())};
    }

    // The merged module maps each input to its own parameter
    bool has_distinct_inputs() const
    {
        auto inputs = tensor_inputs();
        std::unordered_set<instruction_ref> distinct(inputs.begin(), inputs.end());
        return distinct.size() == inputs.size() and
               pointwise_module().get_parameter_names().size() == inputs.size();
    }

    instruction_ref slot() const { return producer->inputs().back(); }

    const module& pointwise_module() const { return *producer->module_inputs().front(); }

    // Every input of the kernel is viewed in the lens it computes in
    bool can_view(const std::vector<std::size_t>& lens) const
    {
        auto op = make_op("reshape_lazy", {{"dims", lens}});
        return std::all_of(producer->inputs().begin(), producer->inputs().end(), [&](auto x) {
            return x->get_shape().lens() == lens or
                   find_concat_past_present::view_shape(x->get_shape(), {op}).has_value();
        });
    }
};

// Two appends of a step writing different caches, merged into one pointwise
// kernel with an output per slot: the slots are packed by make_tuple so each
// output aliases its own cache
struct append_merge
{
    fused_append a;
    fused_append b;

    // The first write moves down to the second producer, which must follow
    // the first append with nothing in between reading the first cache, and
    // neither kernel may read the cache the other writes
    bool ordered() const
    {
        if(reads_buffer(a.producer, buffers(b.cache)))
            return false;
        auto span  = range(std::next(a.dep), b.dep);
        auto after = iterator_for(span);
        if(std::find(after.begin(), after.end(), b.producer) == after.end())
            return false;
        auto bufs_a  = buffers(a.cache);
        auto moved   = range(std::next(a.dep), std::next(b.producer));
        auto between = iterator_for(moved);
        return std::none_of(between.begin(), between.end(), [&](instruction_ref ins) {
            return reads_buffer(ins, bufs_a);
        });
    }

    // The lens both kernels can compute in, preferring the first producer's
    optional<std::vector<std::size_t>> common_lens() const
    {
        const auto& sa = a.producer->get_shape();
        const auto& sb = b.producer->get_shape();
        if(sa.elements() != sb.elements())
            return nullopt;
        if(not a.has_distinct_inputs() or not b.has_distinct_inputs())
            return nullopt;
        if(b.can_view(sa.lens()))
            return sa.lens();
        if(a.can_view(sb.lens()))
            return sb.lens();
        return nullopt;
    }

    static instruction_ref
    view(module& m, instruction_ref at, instruction_ref x, const std::vector<std::size_t>& lens)
    {
        if(x->get_shape().lens() == lens)
            return x;
        return m.insert_instruction(at, make_op("reshape_lazy", {{"dims", lens}}), x);
    }

    static std::vector<instruction_ref> view_all(module& m,
                                                 instruction_ref at,
                                                 const std::vector<instruction_ref>& xs,
                                                 const std::vector<std::size_t>& lens)
    {
        std::vector<instruction_ref> result;
        std::transform(xs.begin(), xs.end(), std::back_inserter(result), [&](instruction_ref x) {
            return view(m, at, x, lens);
        });
        return result;
    }

    void apply(module_pass_manager& mpm, const std::vector<std::size_t>& lens) const
    {
        auto& m    = mpm.get_module();
        auto at    = b.producer;
        auto ins_a = view_all(m, at, a.tensor_inputs(), lens);
        auto ins_b = view_all(m, at, b.tensor_inputs(), lens);
        auto slots = view_all(m, at, {a.slot(), b.slot()}, lens);

        // The second module is appended to a copy of the first, returning the
        // outputs of both
        module pm    = a.pointwise_module();
        auto map_ins = pm.get_ins_param_map(ins_a);
        auto returns = pm.get_returns();
        auto rb      = pm.fuse(b.pointwise_module(), ins_b, &map_ins, nullptr, &to_scalar);
        returns.insert(returns.end(), rb.begin(), rb.end());
        pm.replace_return(returns);
        auto inputs = find_inputs(map_ins, &m, &pm);
        inputs.push_back(m.insert_instruction(at, make_op("make_tuple"), slots));

        auto name = a.pointwise_module().name();
        mpm.rename_module(name, name + ":" + b.pointwise_module().name() + "-deleted");
        auto* merged = mpm.create_module(name, std::move(pm));

        auto v            = a.producer->get_operator().to_value();
        v["output_shape"] = to_value(inputs.back()->get_shape());
        auto fused = m.insert_instruction(at, make_op("gpu::precompile_op", v), inputs, {merged});
        m.replace_instruction(a.dep, m.insert_instruction(at, make_op("identity"), a.cache, fused));
        m.replace_instruction(b.dep, m.insert_instruction(at, make_op("identity"), b.cache, fused));
    }

    bool run(module_pass_manager& mpm) const
    {
        if(not ordered())
            return false;
        auto lens = common_lens();
        if(not lens.has_value())
            return false;
        apply(mpm, *lens);
        return true;
    }
};

// Consecutive fused appends are merged pairwise
void merge_sibling_appends(module_pass_manager& mpm)
{
    optional<fused_append> prev;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        auto cur = fused_append::from_dep(ins);
        if(not cur.has_value())
            continue;
        if(prev.has_value() and append_merge{*prev, *cur}.run(mpm))
            prev = nullopt;
        else
            prev = cur;
    }
}

} // namespace

void fuse_concat_past_present::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm.get_module(), find_concat_past_present{});
    if(merge_appends)
        merge_sibling_appends(mpm);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
