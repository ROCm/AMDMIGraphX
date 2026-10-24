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
#include <migraphx/register_op.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// A size-1 slice of the input along axis at a runtime index, writable by a
// fused producer. concat_past_present silently skips a write whose index is
// outside the axis (eg a decode step past the cache's current length); this
// view-based replacement cannot skip, so an out-of-range index redirects the
// write into a dedicated scratch slot instead, leaving the cache untouched.
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
        check_shapes{inputs, *this}.has(3);
        const auto& s = inputs[0];
        if(axis >= s.ndim())
            MIGRAPHX_THROW("SLICE_AT: axis out of range");
        if(inputs[1].elements() != 1 or not shape::is_integral(inputs[1].type()))
            MIGRAPHX_THROW("SLICE_AT: index must be a single integer");
        auto lens  = s.lens();
        lens[axis] = 1;
        shape result{s.type(), lens, s.strides()};
        if(inputs[2].lens() != result.lens() or inputs[2].type() != result.type())
            MIGRAPHX_THROW("SLICE_AT: scratch input must match the sliced shape");
        return result;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto& s = args[0].get_shape();
        auto idx      = args[1].at<std::int64_t>();
        if(idx < 0 or idx >= static_cast<std::int64_t>(s.lens()[axis]))
        {
            const auto& scratch = args[2];
            return {output_shape, [=] { return scratch.data(); }};
        }
        std::vector<std::size_t> start(s.ndim(), 0);
        start[axis]       = idx;
        auto offset       = s.index(start) * s.type_size();
        const auto& input = args[0];
        return {output_shape, [=] { return input.data() + offset; }};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0, 2}; }
};
MIGRAPHX_REGISTER_OP(slice_at);

namespace {

// concat_past_present fixes the (batch, kv_heads, seq, head) layout; seq is the append axis
constexpr std::size_t seq_axis = 2;

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
            auto slot_s = slot_shape(cur, cache);
            auto scratch =
                m.insert_instruction(ins, make_op("allocate", {{"shape", to_value(slot_s)}}));
            view = m.insert_instruction(ins,
                                        make_op("gpu::slice_at", {{"axis", seq_axis}}),
                                        cache,
                                        load_scalar(m, ins, slk),
                                        scratch);
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

} // namespace

void fuse_concat_past_present::apply(module& m) const
{
    match::find_matches(m, find_concat_past_present{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
