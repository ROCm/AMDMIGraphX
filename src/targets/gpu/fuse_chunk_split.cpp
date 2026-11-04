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

#include <migraphx/gpu/fuse_chunk_split.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <algorithm>
#include <deque>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Walks the chunks of a linear recurrence with the state resident:
//   r  = [w_i S] ; u = read(u0_i, w_i S) or u0_i
//   o_i = out(pu0_i, q_i S) or q_i S
//   S  = step(S, gamma_i, upd_i),  upd_i = kt_i u or a precomputed view
// Each per-chunk operand family is one view of chunk 0 over its base input
// plus the element stride to the next chunk. The submodules are the step,
// read and out pointwise bodies in that order, each with its inputs in the
// order listed above.
struct chunk_scan
{
    shape output_shape;
    shape state_shape;
    std::vector<shape> views;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> deltas;
    std::size_t chunks  = 0;
    bool has_state      = false; // S_0 input, else the walk starts from zero
    bool has_update_dot = false; // kt and u0 families, else a precomputed update family
    bool has_read       = false; // w family and the read module
    bool has_gamma      = false;
    bool has_out        = false; // pu0 family and the out module

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"),
                    f(self.state_shape, "state_shape"),
                    f(self.views, "views"),
                    f(self.offsets, "offsets"),
                    f(self.deltas, "deltas"),
                    f(self.chunks, "chunks"),
                    f(self.has_state, "has_state"),
                    f(self.has_update_dot, "has_update_dot"),
                    f(self.has_read, "has_read"),
                    f(self.has_gamma, "has_gamma"),
                    f(self.has_out, "has_out"));
    }

    std::string name() const { return "gpu::chunk_scan"; }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        if(inputs.size() != views.size() + (has_state ? 1 : 0))
            MIGRAPHX_THROW("chunk_scan: expected " + std::to_string(views.size()) +
                           " family inputs" + (has_state ? " and the state" : ""));
        if(mods.size() != 1 + (has_read ? 1 : 0) + (has_out ? 1 : 0))
            MIGRAPHX_THROW("chunk_scan: wrong number of submodules");
        return shape{{output_shape, state_shape}};
    }
};
MIGRAPHX_REGISTER_OP(chunk_scan);

namespace {

// Where a chain of views lands in its base buffer
struct view_origin
{
    instruction_ref base;
    shape view;
    std::size_t offset = 0;
};

// Reshape copies unless its input is standard, so only then is it a view
bool is_view(instruction_ref ins)
{
    static const std::vector<std::string> views = {
        "slice", "squeeze", "unsqueeze", "transpose", "multibroadcast", "broadcast"};
    if(contains(views, ins->name()))
        return true;
    return ins->name() == "reshape" and ins->inputs().front()->get_shape().standard();
}

view_origin origin_of(instruction_ref ins)
{
    view_origin r{ins, ins->get_shape(), 0};
    while(is_view(r.base))
    {
        if(r.base->name() == "slice")
        {
            auto op          = any_cast<op::slice>(r.base->get_operator());
            const auto& in_s = r.base->inputs().front()->get_shape();
            // compute_offset is in bytes
            r.offset += op.compute_offset(in_s) / in_s.type_size();
        }
        r.base = r.base->inputs().front();
    }
    return r;
}

bool is_dot(instruction_ref ins) { return ins->name() == "dot"; }

// The instruction a chain of views reads from
instruction_ref strip_views(instruction_ref ins)
{
    while(is_view(ins))
        ins = ins->inputs().front();
    return ins;
}

// The instruction a chain of single-use views feeds
instruction_ref follow_views(instruction_ref ins)
{
    while(ins->outputs().size() == 1 and is_view(ins->outputs().front()))
        ins = ins->outputs().front();
    return ins;
}

// The dots reading ins as their right operand, directly or through views
void collect_dot_readers(instruction_ref src, std::vector<instruction_ref>& readers)
{
    for(auto out : src->outputs())
    {
        if(is_dot(out) and out->inputs().at(1) == src)
            readers.push_back(out);
        else if(is_view(out))
            collect_dot_readers(out, readers);
    }
}

std::vector<instruction_ref> dot_readers(instruction_ref ins)
{
    std::vector<instruction_ref> readers;
    collect_dot_readers(ins, readers);
    return readers;
}

// Whether ins is read as the right operand of a dot: the state is, and the
// decay and precomputed updates are not
bool read_by_dot(instruction_ref ins) { return not dot_readers(ins).empty(); }

// One transition S' = step(S, [gamma_i], upd_i) with the instructions it
// consumes from the chunk's operand families
struct chunk_step
{
    instruction_ref state_in;
    instruction_ref state_out;
    instruction_ref upd; // the precomputed view, or the dot(kt, u)
    std::optional<instruction_ref> gamma;
    std::optional<instruction_ref> kt;
    std::optional<instruction_ref> u0;
    std::optional<instruction_ref> w;
    std::optional<instruction_ref> read; // u = read(u0, dot(w, S))
    std::optional<instruction_ref> ws;   // the read's input that comes from dot(w, S)
};

std::optional<chunk_step> match_step(instruction_ref pw)
{
    if(pw->name() != "pointwise")
        return std::nullopt;
    const auto& inputs = pw->inputs();
    if(inputs.size() < 2 or inputs.size() > 3)
        return std::nullopt;
    chunk_step step;
    step.state_out = pw;
    // the state is the input other dots read; of the rest, the update is the
    // dot, or without one the view that is not a broadcast, and the decay is
    // the broadcast view
    auto state_it = std::find_if(inputs.begin(), inputs.end(), read_by_dot);
    if(state_it == inputs.end())
        return std::nullopt;
    step.state_in = *state_it;
    std::vector<instruction_ref> rest;
    std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(rest), [&](auto in) {
        return in != step.state_in;
    });
    if(std::any_of(rest.begin(), rest.end(), [](auto in) {
           return in->outputs().size() != 1 or not(is_dot(in) or is_view(in));
       }))
        return std::nullopt;
    auto upd_it = std::find_if(rest.begin(), rest.end(), is_dot);
    if(upd_it == rest.end())
        upd_it = std::find_if(
            rest.begin(), rest.end(), [](auto in) { return not in->get_shape().broadcasted(); });
    if(upd_it == rest.end())
        return std::nullopt;
    step.upd = *upd_it;
    rest.erase(upd_it);
    if(rest.size() > 1)
        return std::nullopt;
    if(not rest.empty())
    {
        if(not is_view(rest.front()) or not rest.front()->get_shape().broadcasted())
            return std::nullopt;
        step.gamma = rest.front();
    }
    if(is_dot(step.upd))
    {
        step.kt = step.upd->inputs().at(0);
        auto u  = strip_views(step.upd->inputs().at(1));
        if(u->name() == "pointwise" and u->inputs().size() == 2)
        {
            // u = read(u0, dot(w, S)), possibly through views on either side
            auto ws = std::find_if(u->inputs().begin(), u->inputs().end(), [&](auto in) {
                auto d = strip_views(in);
                return is_dot(d) and strip_views(d->inputs().at(1)) == step.state_in and
                       d->outputs().size() == 1;
            });
            if(ws == u->inputs().end())
                return std::nullopt;
            step.read = u;
            step.ws   = *ws;
            step.w    = strip_views(*ws)->inputs().at(0);
            step.u0   = u->inputs().at(*ws == u->inputs().front() ? 1 : 0);
        }
        else
        {
            step.u0 = step.upd->inputs().at(1);
        }
    }
    return step;
}

// A per-chunk operand family: every chunk is the same view of one base,
// each chunk a constant number of elements further in
struct family
{
    instruction_ref base;
    shape view;
    std::size_t offset = 0;
    std::size_t delta  = 0;
};

std::optional<family> make_family(const std::vector<std::optional<instruction_ref>>& chunks)
{
    std::optional<family> f;
    std::optional<std::size_t> first; // index and offset of the first present chunk
    std::size_t first_offset = 0;
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        if(not chunks[i].has_value())
            continue;
        auto o = origin_of(*chunks[i]);
        if(not f.has_value())
        {
            f            = family{o.base, o.view, o.offset, 0};
            first        = i;
            first_offset = o.offset;
            continue;
        }
        if(o.base != f->base or o.view != f->view or o.offset < first_offset)
            return std::nullopt;
        auto delta = (o.offset - first_offset) / (i - *first);
        if(delta * (i - *first) != o.offset - first_offset)
            return std::nullopt;
        if(i - *first > 1 and delta != f->delta)
            return std::nullopt;
        f->delta = delta;
    }
    if(not f.has_value())
        return std::nullopt;
    // chunk 0 may be absent when the walk starts from zero
    if(f->offset < *first * f->delta)
        return std::nullopt;
    f->offset -= *first * f->delta;
    return f;
}

// The matched loop over n chunks: the state entering each chunk, the
// transition of each chunk, the output read q_i S of each chunk and its
// out pointwise, and the concat assembling the outputs. Entries are absent
// for chunk 0 when the walk starts from zero.
struct chunk_loop
{
    std::size_t chunks = 0;
    bool has_state     = false;
    instruction_ref final_state;
    instruction_ref concat;
    std::vector<std::optional<instruction_ref>> states;
    std::vector<std::optional<chunk_step>> steps;
    std::vector<std::optional<instruction_ref>> reads;
    std::vector<std::optional<instruction_ref>> outs;
    std::vector<std::optional<instruction_ref>> pu0s;   // the out pointwise's other input
    std::vector<std::optional<instruction_ref>> qs_ins; // the out pointwise's input from the read
};

template <class F>
std::vector<std::optional<instruction_ref>> gather(const chunk_loop& loop, F f)
{
    std::vector<std::optional<instruction_ref>> result(loop.chunks);
    std::transform(loop.steps.begin(), loop.steps.end(), result.begin(), [&](const auto& step) {
        return step.has_value() ? f(*step) : std::optional<instruction_ref>{};
    });
    return result;
}

// Follow the state from the first transition of a chain
std::optional<instruction_ref> output_read(instruction_ref state,
                                           const std::optional<chunk_step>& step)
{
    std::optional<instruction_ref> result;
    for(auto out : dot_readers(state))
    {
        if(step.has_value() and step->ws.has_value() and strip_views(*step->ws) == out)
            continue;
        if(result.has_value())
        {
            return std::nullopt;
        }
        result = out;
    }
    return result;
}

// The output of one chunk: the read q_i S of its state, the out pointwise
// it may pass through, and the concat joining it
struct chunk_output
{
    instruction_ref read;
    std::optional<instruction_ref> out;
    std::optional<instruction_ref> qs_in;
    std::optional<instruction_ref> pu0;
    instruction_ref concat;
};

std::optional<chunk_output> trace_output(instruction_ref state,
                                         const std::optional<chunk_step>& step)
{
    auto read = output_read(state, step);
    if(not read.has_value() or (*read)->outputs().size() != 1)
        return std::nullopt;
    chunk_output result;
    result.read = *read;
    auto user   = follow_views(*read);
    if(user->outputs().size() != 1)
        return std::nullopt;
    user = user->outputs().front();
    if(user->name() == "pointwise" and user->inputs().size() == 2 and user->outputs().size() == 1)
    {
        const auto& ins = user->inputs();
        bool first      = strip_views(ins.front()) == *read;
        result.out      = user;
        result.qs_in    = ins.at(first ? 0 : 1);
        result.pu0      = ins.at(first ? 1 : 0);
        user            = follow_views(user);
        if(user->outputs().size() != 1)
            return std::nullopt;
        user = user->outputs().front();
    }
    if(user->name() != "concat")
        return std::nullopt;
    result.concat = user;
    return result;
}

// The step pointwises of a module by their incoming state
using step_map = std::unordered_map<instruction_ref, chunk_step>;

std::optional<chunk_loop> trace_loop(instruction_ref first, const step_map& steps)
{
    std::vector<chunk_step> chain;
    std::optional<instruction_ref> concat;
    auto it = steps.find(first);
    while(it != steps.end())
    {
        // a transition whose chunk joins another concat starts the next
        // recurrence
        auto out = trace_output(it->second.state_in, it->second);
        auto c   = out.has_value() ? std::optional<instruction_ref>{out->concat}
                                   : std::optional<instruction_ref>{};
        if(chain.empty())
            concat = c;
        else if(c != concat)
            break;
        chain.push_back(it->second);
        auto out_state = it->second.state_out;
        it             = std::find_if(steps.begin(), steps.end(), [&](const auto& p) {
            return p.second.state_in == out_state;
        });
    }
    const auto& head = chain.front();
    bool update_dot  = head.kt.has_value();
    if(std::any_of(chain.begin(), chain.end(), [&](const chunk_step& step) {
           return step.kt.has_value() != update_dot or
                  step.read.has_value() != head.read.has_value() or
                  step.gamma.has_value() != head.gamma.has_value();
       }))
        return std::nullopt;

    // Without a past state chunk 0 has no transition: S_1 is the update of
    // chunk 0, which is dot(kt_0, u0_0) or the precomputed update itself
    chunk_loop loop;
    auto zero_start = [&] {
        if(update_dot)
        {
            if(not is_dot(head.state_in))
                return false;
            std::vector<std::optional<instruction_ref>> kts{head.state_in->inputs().at(0)};
            std::vector<std::optional<instruction_ref>> u0s{head.state_in->inputs().at(1)};
            std::transform(chain.begin(),
                           chain.end(),
                           std::back_inserter(kts),
                           [](const auto& step) { return step.kt; });
            std::transform(chain.begin(),
                           chain.end(),
                           std::back_inserter(u0s),
                           [](const auto& step) { return step.u0; });
            return make_family(kts).has_value() and make_family(u0s).has_value();
        }
        std::vector<std::optional<instruction_ref>> upds{head.state_in};
        std::transform(chain.begin(), chain.end(), std::back_inserter(upds), [](const auto& step) {
            return std::optional<instruction_ref>{step.upd};
        });
        return is_view(head.state_in) and make_family(upds).has_value();
    }();
    loop.has_state   = not zero_start;
    loop.chunks      = chain.size() + (zero_start ? 1 : 0);
    loop.final_state = chain.back().state_out;
    loop.states.resize(loop.chunks);
    loop.steps.resize(loop.chunks);
    loop.reads.resize(loop.chunks);
    loop.outs.resize(loop.chunks);
    loop.pu0s.resize(loop.chunks);
    loop.qs_ins.resize(loop.chunks);
    std::size_t base = zero_start ? 1 : 0;
    if(not zero_start)
        loop.states[0] = head.state_in;
    for(std::size_t i = 0; i < chain.size(); i++)
    {
        loop.steps[base + i] = chain[i];
        if(base + i + 1 < loop.chunks)
            loop.states[base + i + 1] = chain[i].state_out;
        if(zero_start and i == 0)
            loop.states[1] = chain[0].state_in;
    }
    return loop;
}

// The output read of a state: dot(q_i, S) that is not the read feeding the
// chunk's transition
// Find the reads q_i S, their out pointwise and the concat joining them,
// looking through the views the simplifier leaves between them
bool trace_outputs(chunk_loop& loop)
{
    std::optional<instruction_ref> concat;
    bool per_step_out = false;
    for(std::size_t i = 0; i < loop.chunks; i++)
    {
        if(not loop.states[i].has_value())
            continue;
        auto out = trace_output(*loop.states[i], loop.steps[i]);
        if(not out.has_value() or (concat.has_value() and *concat != out->concat))
            return false;
        loop.reads[i]  = out->read;
        loop.outs[i]   = out->out;
        loop.qs_ins[i] = out->qs_in;
        loop.pu0s[i]   = out->pu0;
        per_step_out   = per_step_out or out->out.has_value();
        concat         = out->concat;
    }
    if(not concat.has_value())
        return false;
    loop.concat = *concat;
    // every chunk output in order, chunk 0 of a zero start being its pu0 view
    const auto& joined = loop.concat->inputs();
    if(joined.size() != loop.chunks)
    {
        return false;
    }
    for(std::size_t i = 0; i < loop.chunks; i++)
    {
        if(loop.states[i].has_value())
        {
            auto expected = loop.outs[i].has_value() ? *loop.outs[i] : *loop.reads[i];
            if(strip_views(joined[i]) != expected or loop.outs[i].has_value() != per_step_out)
                return false;
        }
        else if(per_step_out and is_view(joined[i]))
        {
            loop.pu0s[i] = joined[i];
        }
        else
        {
            return false;
        }
    }
    // the chunks are joined along the rows of the (.., n * rows, dv) output
    auto out_lens = loop.concat->get_shape().lens();
    auto rows     = (*loop.reads.at(loop.has_state ? 0 : 1))->get_shape().lens();
    return loop.concat->get_shape().standard() and
           loop.concat->get_shape().elements() ==
               rows.at(rows.size() - 2) * rows.back() * loop.chunks *
                   (loop.final_state->get_shape().elements() /
                    (loop.final_state->get_shape().lens().at(loop.final_state->get_shape().ndim() -
                                                             2) *
                     loop.final_state->get_shape().lens().back()));
}

// A pointwise body with its parameters retyped to the accumulation type and
// the converts the fusion pulled in at its boundaries dropped, so the chunks'
// bodies compare equal; the kernel converts at its loads and stores instead
module canonical_body(const module& pm, shape::type_t type)
{
    module m;
    std::unordered_map<instruction_ref, instruction_ref> map;
    for(const auto& name : pm.get_parameter_names())
        map[pm.get_parameter(name)] = m.add_parameter(name, shape{type});
    auto outs = m.add_instructions(&pm, &map);
    std::transform(outs.begin(), outs.end(), outs.begin(), [](auto out) {
        while(out->name() == "convert")
            out = out->inputs().front();
        return out;
    });
    m.add_return(outs);
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "convert" and
           ins->get_shape().type() == ins->inputs().front()->get_shape().type())
            m.replace_instruction(ins, ins->inputs().front());
    }
    run_passes(m, {dead_code_elimination{}});
    return m;
}

// The canonical body of a pointwise as a new module whose parameters follow
// `roles`, the inputs of the pointwise in the order the kernel passes them
module_ref copy_pointwise(module_pass_manager& mpm,
                          const std::string& name,
                          instruction_ref pw,
                          const std::vector<instruction_ref>& roles,
                          shape::type_t type)
{
    auto body     = canonical_body(*pw->module_inputs().front(), type);
    module_ref rm = mpm.create_module(name);
    rm->set_bypass();
    std::unordered_map<instruction_ref, instruction_ref> map;
    for(std::size_t k = 0; k < roles.size(); k++)
    {
        auto j =
            std::find(pw->inputs().begin(), pw->inputs().end(), roles[k]) - pw->inputs().begin();
        auto param = body.get_parameter("x" + std::to_string(j));
        map[param] = rm->add_parameter("x" + std::to_string(k), param->get_shape());
    }
    rm->add_return(rm->add_instructions(&body, &map));
    return rm;
}

bool same_modules(const std::vector<std::optional<instruction_ref>>& pws, shape::type_t type)
{
    std::optional<module> first;
    for(const auto& pw : pws)
    {
        if(not pw.has_value())
            continue;
        auto body = canonical_body(*(*pw)->module_inputs().front(), type);
        if(not first.has_value())
            first = body;
        else if(not(*first == body))
            return false;
    }
    return true;
}

std::vector<instruction_ref> step_roles(const chunk_step& step)
{
    std::vector<instruction_ref> roles{step.state_in};
    if(step.gamma.has_value())
        roles.push_back(*step.gamma);
    roles.push_back(step.upd);
    return roles;
}

// Drop leading unit axes until the view has the state's rank; the simplifier
// leaves the chunk outputs with an unsqueezed chunk axis
std::optional<shape> squeeze_to(const shape& view, std::size_t ndim)
{
    auto lens    = view.lens();
    auto strides = view.strides();
    while(lens.size() > ndim)
    {
        auto it = std::find(lens.rbegin() + 2, lens.rend(), 1);
        if(it == lens.rend())
            return std::nullopt;
        auto axis = std::distance(it, lens.rend()) - 1;
        lens.erase(lens.begin() + axis);
        strides.erase(strides.begin() + axis);
    }
    if(lens.size() != ndim)
        return std::nullopt;
    return shape{view.type(), lens, strides};
}

std::size_t position(const module& m, instruction_ref ins) { return std::distance(m.begin(), ins); }

using chunk_family = std::vector<std::optional<instruction_ref>>;

// The op's flags and shapes: the kernel writes (lead..., n * rows, dv); the
// concat's own layout is a reshape of that
chunk_scan scan_op(const chunk_loop& loop, const chunk_step& head)
{
    chunk_scan op;
    op.chunks         = loop.chunks;
    op.has_state      = loop.has_state;
    op.has_update_dot = head.kt.has_value();
    op.has_read       = head.read.has_value();
    op.has_gamma      = head.gamma.has_value();
    op.has_out        = std::any_of(
        loop.outs.begin(), loop.outs.end(), [](const auto& o) { return o.has_value(); });
    op.state_shape                   = loop.final_state->get_shape();
    auto out_lens                    = op.state_shape.lens();
    auto q_lens                      = (*loop.reads.at(loop.has_state ? 0 : 1))->get_shape().lens();
    out_lens.at(out_lens.size() - 2) = q_lens.at(q_lens.size() - 2) * loop.chunks;
    op.output_shape                  = shape{loop.concat->get_shape().type(), out_lens};
    return op;
}

// The per-chunk operands of each family in the op's input order
std::vector<chunk_family> operand_families(const chunk_loop& loop, const chunk_scan& op)
{
    std::vector<chunk_family> families;
    if(op.has_read)
        families.push_back(gather(loop, [](const chunk_step& s) { return *s.w; }));
    auto q = loop.reads;
    std::transform(q.begin(), q.end(), q.begin(), [](auto r) {
        return r.has_value() ? std::optional<instruction_ref>{(*r)->inputs().at(0)} : r;
    });
    families.push_back(q);
    if(op.has_update_dot)
    {
        auto kts = gather(loop, [](const chunk_step& s) { return *s.kt; });
        auto u0s = gather(loop, [](const chunk_step& s) { return *s.u0; });
        if(not loop.has_state)
        {
            kts[0] = (*loop.states[1])->inputs().at(0);
            u0s[0] = (*loop.states[1])->inputs().at(1);
        }
        families.push_back(kts);
        families.push_back(u0s);
    }
    else
    {
        auto upds = gather(loop, [](const chunk_step& s) { return s.upd; });
        if(not loop.has_state)
            upds[0] = loop.states[1];
        families.push_back(upds);
    }
    if(op.has_gamma)
        families.push_back(gather(loop, [](const chunk_step& s) { return *s.gamma; }));
    if(op.has_out)
        families.push_back(loop.pu0s);
    return families;
}

// Encode a family as a view of its base in the op and return the base; the
// leading axes are the problem axes, shared with the state
std::optional<instruction_ref> add_family(chunk_scan& op, const chunk_family& chunks)
{
    auto f = make_family(chunks);
    if(not f.has_value())
        return std::nullopt;
    auto view = squeeze_to(f->view, op.state_shape.ndim());
    if(not view.has_value())
        return std::nullopt;
    if(not std::equal(
           view->lens().begin(), std::prev(view->lens().end(), 2), op.state_shape.lens().begin()))
        return std::nullopt;
    op.views.push_back(*view);
    op.offsets.push_back(f->offset);
    op.deltas.push_back(f->delta);
    return f->base;
}

// Every chunk's step, read and out body must be the same pointwise
bool same_bodies(const chunk_loop& loop, const chunk_scan& op, shape::type_t type)
{
    chunk_family transitions(loop.chunks);
    std::transform(loop.steps.begin(), loop.steps.end(), transitions.begin(), [](const auto& s) {
        return s.has_value() ? std::optional<instruction_ref>{s->state_out}
                             : std::optional<instruction_ref>{};
    });
    auto reads = gather(
        loop, [](const chunk_step& s) { return s.read.has_value() ? *s.read : s.state_out; });
    return same_modules(transitions, type) and (not op.has_read or same_modules(reads, type)) and
           (not op.has_out or same_modules(loop.outs, type));
}

// The step, read and out submodules in the kernel's order
std::vector<module_ref> scan_modules(module_pass_manager& mpm,
                                     const chunk_loop& loop,
                                     const chunk_scan& op,
                                     const std::string& prefix,
                                     shape::type_t type)
{
    const auto& head = std::find_if(loop.steps.begin(), loop.steps.end(), [](const auto& s) {
                           return s.has_value();
                       })->value();
    std::vector<module_ref> mods;
    mods.push_back(copy_pointwise(mpm, prefix + ":step", head.state_out, step_roles(head), type));
    if(op.has_read)
        mods.push_back(
            copy_pointwise(mpm, prefix + ":read", *head.read, {*head.u0, *head.ws}, type));
    if(op.has_out)
    {
        auto it = std::find_if(
            loop.outs.begin(), loop.outs.end(), [](const auto& o) { return o.has_value(); });
        auto i = it - loop.outs.begin();
        mods.push_back(
            copy_pointwise(mpm, prefix + ":out", **it, {*loop.pu0s[i], *loop.qs_ins[i]}, type));
    }
    return mods;
}

bool fuse_loop(module_pass_manager& mpm, const chunk_loop& loop, const std::string& prefix)
{
    auto& m          = mpm.get_module();
    const auto& head = std::find_if(loop.steps.begin(), loop.steps.end(), [](const auto& s) {
                           return s.has_value();
                       })->value();
    auto op          = scan_op(loop, head);
    std::vector<instruction_ref> inputs;
    if(op.has_state)
        inputs.push_back(*loop.states[0]);
    for(const auto& chunks : operand_families(loop, op))
    {
        auto base = add_family(op, chunks);
        if(not base.has_value())
            return false;
        inputs.push_back(*base);
    }
    // every input must be computed before the loop is replaced
    auto at = position(m, loop.concat);
    if(std::any_of(inputs.begin(), inputs.end(), [&](auto in) { return position(m, in) >= at; }))
        return false;
    // the state accumulates in the type of the chunk operands
    const auto type = inputs.back()->get_shape().type();
    if(not same_bodies(loop, op, type))
        return false;

    auto mods  = scan_modules(mpm, loop, op, prefix, type);
    auto scan  = m.insert_instruction(loop.concat, op, inputs, mods);
    auto out   = m.insert_instruction(loop.concat, make_op("get_tuple_elem", {{"index", 0}}), scan);
    auto state = m.insert_instruction(loop.concat, make_op("get_tuple_elem", {{"index", 1}}), scan);
    if(loop.concat->get_shape().lens() != op.output_shape.lens())
        out = m.insert_instruction(
            loop.concat, make_op("reshape", {{"dims", loop.concat->get_shape().lens()}}), out);
    m.replace_instruction(loop.concat, out);
    m.replace_instruction(loop.final_state, state);
    return true;
}

} // namespace

void fuse_chunk_split::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    step_map steps;
    for(auto ins : iterator_for(m))
    {
        auto step = match_step(ins);
        if(step.has_value())
            steps.emplace(ins, *step);
    }
    // chains start where the incoming state is not another transition's
    // output; walk them in module order so the fusions are deterministic
    std::deque<instruction_ref> heads;
    auto im = iterator_for(m);
    transform_if(
        im.begin(),
        im.end(),
        std::back_inserter(heads),
        [&](auto ins) {
            auto it = steps.find(ins);
            return it != steps.end() and steps.count(it->second.state_in) == 0;
        },
        [](auto ins) { return ins; });
    std::vector<chunk_loop> loops;
    while(not heads.empty())
    {
        auto loop = trace_loop(heads.front(), steps);
        heads.pop_front();
        if(not loop.has_value())
            continue;
        // a recurrence fed by this one's final state is traced next
        auto next = std::find_if(steps.begin(), steps.end(), [&](const auto& p) {
            return p.second.state_in == loop->final_state;
        });
        if(next != steps.end())
            heads.push_back(next->first);
        if(trace_outputs(*loop))
            loops.push_back(*loop);
    }
    // a recurrence fed by another's final state is fused first, so replacing
    // that state rewires its scan; the names follow the discovery order
    std::reverse(loops.begin(), loops.end());
    auto n = loops.size();
    for(const auto& loop : loops)
    {
        n--;
        fuse_loop(mpm, loop, m.name() + ":chunk_scan" + std::to_string(n));
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
