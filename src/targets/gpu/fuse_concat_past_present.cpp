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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern std::string hip_error(int error);

// Copy a single scalar from the gpu to the host, synchronizing the stream so
// later host-side view computations can read it
struct load_scalar
{
    std::string name() const { return "gpu::load_scalar"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        if(inputs.front().elements() != 1)
            MIGRAPHX_THROW("LOAD_SCALAR: input must have a single element");
        return {inputs.front().type(), inputs.front().lens()};
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        copy_from_gpu(ctx, args.front(), result);
        // Spin on the stream instead of synchronizing to avoid the scheduler
        // wake latency; only the scalar's producer can still be pending here
        for(;;)
        {
            auto status = hipStreamQuery(ctx.get_stream().get());
            if(status == hipSuccess)
                break;
            if(status != hipErrorNotReady)
                MIGRAPHX_THROW("LOAD_SCALAR: stream query failed: " + hip_error(status));
        }
        return result;
    }
};
MIGRAPHX_REGISTER_OP(load_scalar);

// A size-1 slice of the input along axis at a runtime index. The index is
// clamped into range: concat_past_present skips out-of-range writes, a view
// must instead target a valid slot.
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
        if(axis >= s.lens().size())
            MIGRAPHX_THROW("SLICE_AT: axis out of range");
        if(inputs.back().elements() != 1)
            MIGRAPHX_THROW("SLICE_AT: index must have a single element");
        auto lens  = s.lens();
        lens[axis] = 1;
        return {s.type(), lens, s.strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        std::ptrdiff_t idx = 0;
        args[1].visit([&](auto v) { idx = static_cast<std::ptrdiff_t>(v.front()); });
        const auto& s = args[0].get_shape();
        idx = std::clamp<std::ptrdiff_t>(idx, 0, static_cast<std::ptrdiff_t>(s.lens()[axis]) - 1);
        auto offset = idx * s.strides()[axis] * s.type_size();
        auto input  = args[0];
        return {output_shape, [=] { return input.data() + offset; }};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};
MIGRAPHX_REGISTER_OP(slice_at);

// Returns the first input; the remaining inputs are dependencies that write
// into the first input's buffer
struct depends_on
{
    std::string name() const { return "gpu::depends_on"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
            MIGRAPHX_THROW("DEPENDS_ON: missing inputs");
        return inputs.front();
    }

    argument compute(const shape&, std::vector<argument> args) const { return args.front(); }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};
MIGRAPHX_REGISTER_OP(depends_on);

namespace {

const std::unordered_set<std::string>& reorder_view_ops()
{
    static const std::unordered_set<std::string> names = {
        "unsqueeze", "squeeze", "transpose", "reshape_lazy"};
    return names;
}

std::string precompile_name(instruction_ref ins)
{
    if(ins->name() != "gpu::precompile_op")
        return "";
    auto v = ins->get_operator().to_value();
    return from_value<operation>(v.at("op")).name();
}

// Walk from the concat input through single-use reorder views to the kernel
// that produces it
instruction_ref find_producer(instruction_ref cur)
{
    auto x = cur;
    while(contains(reorder_view_ops(), x->name()))
    {
        if(x->outputs().size() != 1)
            return x;
        x = x->inputs().front();
    }
    return x;
}

// Ensure x is defined before anchor; parameters can always be moved up since
// they have no dependencies
bool ensure_defined_before(module& m,
                           std::unordered_map<instruction_ref, std::size_t>& pos,
                           instruction_ref x,
                           instruction_ref anchor)
{
    if(pos.at(x) <= pos.at(anchor))
        return true;
    if(x->name() != "@param")
        return false;
    m.move_instruction(x, anchor);
    pos[x] = pos.at(anchor);
    return true;
}

} // namespace

void fuse_concat_past_present::apply(module& m) const
{
    std::unordered_map<instruction_ref, std::size_t> pos;
    std::size_t n = 0;
    for(auto ins : iterator_for(m))
        pos[ins] = n++;

    std::unordered_map<instruction_ref, instruction_ref> scalars;
    for(auto ins : iterator_for(m))
    {
        if(precompile_name(ins) != "concat_past_present")
            continue;
        if(ins->inputs().size() != 3)
            continue;
        auto cur   = ins->inputs()[0];
        auto slk   = ins->inputs()[1];
        auto cache = ins->inputs()[2];

        auto producer = find_producer(cur);
        auto pname    = precompile_name(producer);
        if(pname != "pointwise" and pname != "fused_concat")
            continue;
        if(producer->outputs().size() != 1)
            continue;
        auto alloc = producer->inputs().back();
        if(alloc->name() != "allocate" and alloc->name() != "hip::allocate")
            continue;
        if(alloc->outputs().size() != 1)
            continue;

        // The producer must write the same elements in the same memory order
        // that concat_past_present would read
        const auto& ps = producer->get_shape();
        const auto& cs = cur->get_shape();
        const auto& ks = cache->get_shape();
        if(ps.type() != cs.type() or ps.lens() != cs.lens())
            continue;
        if(not ps.standard() or not cs.standard() or not ks.standard())
            continue;
        const auto& lens  = cs.lens();
        const auto& klens = ks.lens();
        if(lens.size() != 4 or klens.size() != 4)
            continue;
        if(lens[0] != klens[0] or lens[1] != klens[1] or lens[3] != klens[3])
            continue;
        auto seq = lens[2];
        if(seq > klens[2])
            continue;

        // Inputs of the view must come before the producer
        if(not ensure_defined_before(m, pos, cache, producer))
            continue;

        instruction_ref view;
        if(seq == 1)
        {
            // Decode appends at a device-computed position: read it on the
            // host once and slice the cache at that offset
            if(lens[0] != 1 or slk->get_shape().elements() != 1)
                continue;
            if(not ensure_defined_before(m, pos, slk, producer))
                continue;
            auto it = scalars.find(slk);
            if(it == scalars.end())
            {
                auto scalar = m.insert_instruction(producer, make_op("gpu::load_scalar"), slk);
                it          = scalars.emplace(slk, scalar).first;
            }
            view = m.insert_instruction(
                producer, make_op("gpu::slice_at", {{"axis", 2}}), cache, it->second);
        }
        else
        {
            // Prompt appends at position zero, which is a static view
            view = m.insert_instruction(
                producer,
                make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {seq}}}),
                cache);
        }

        auto v            = producer->get_operator().to_value();
        v["output_shape"] = to_value(view->get_shape());
        auto new_inputs   = producer->inputs();
        new_inputs.back() = view;
        m.replace_instruction(
            producer, make_op("gpu::precompile_op", v), new_inputs, producer->module_inputs());
        m.replace_instruction(ins, make_op("gpu::depends_on"), {cache, producer});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
