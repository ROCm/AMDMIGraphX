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
#include <migraphx/erase.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <algorithm>
#include <cassert>
#include <deque>
#include <optional>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Walks the chunks of a linear recurrence with the state resident:
//   u   = read(u0_i, w_i S) or u0_i
//   o_i = out(pu0_i, q_i S) or q_i S
//   S   = step(S, gamma_i, upd_i),  upd_i = kt_i u or a precomputed view
// Each per-chunk operand family is one view of chunk 0 over its base input
// plus the element stride to the next chunk; `families` names the present
// ones in the op's input order (w, q, kt and u0 or upd, gamma, pu0), after
// the initial state when there is one. The submodules are the step, read
// and out pointwise bodies in that order, each with its inputs in the order
// listed above.
struct chunk_scan
{
    shape output_shape;
    shape state_shape;
    std::vector<std::string> families;
    std::vector<shape> views;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> deltas;
    std::size_t chunks = 0;
    bool has_state     = false; // S_0 input, else the walk starts from zero

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"),
                    f(self.state_shape, "state_shape"),
                    f(self.families, "families"),
                    f(self.views, "views"),
                    f(self.offsets, "offsets"),
                    f(self.deltas, "deltas"),
                    f(self.chunks, "chunks"),
                    f(self.has_state, "has_state"));
    }

    std::string name() const { return "gpu::chunk_scan"; }

    bool has(const std::string& role) const { return contains(families, role); }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        const std::size_t state_inputs = has_state ? 1 : 0;
        check_shapes{inputs, *this}.has(families.size() + state_inputs);
        if(views.size() != families.size() or offsets.size() != families.size() or
           deltas.size() != families.size())
            MIGRAPHX_THROW("chunk_scan: views, offsets and deltas must have one entry per family");
        if(chunks == 0 or state_shape.ndim() < 2)
            MIGRAPHX_THROW("chunk_scan: needs at least one chunk and a matrix state");
        if(mods.size() != 1 + (has("w") ? 1 : 0) + (has("pu0") ? 1 : 0))
            MIGRAPHX_THROW("chunk_scan: wrong number of submodules");
        // every chunk of a family lies inside its input
        for(std::size_t k = 0; k < families.size(); k++)
        {
            if(offsets[k] + (chunks - 1) * deltas[k] + views[k].element_space() >
               inputs[k + state_inputs].element_space())
                MIGRAPHX_THROW("chunk_scan: family " + families[k] + " reads past its input");
        }
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
    const auto name = ins->name();
    if(contains(views, name))
        return true;
    return name == "reshape" and ins->inputs().front()->get_shape().standard();
}

view_origin origin_of(instruction_ref ins)
{
    view_origin r{ins, ins->get_shape(), 0};
    while(is_view(r.base))
    {
        if(r.base->name() == "slice")
        {
            const auto& op   = any_cast<op::slice>(r.base->get_operator());
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

// Whether ins is read as the right operand of a dot, directly or through
// views: the state is, and the decay and precomputed updates are not
bool read_by_dot(instruction_ref ins)
{
    return std::any_of(ins->outputs().begin(), ins->outputs().end(), [&](auto out) {
        return (is_dot(out) and out->inputs().at(1) == ins) or (is_view(out) and read_by_dot(out));
    });
}

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
    // the state is the input a dot reads; of the rest, the update is the dot,
    // or without one the view that is not a broadcast, and the decay is the
    // broadcast view
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
                return is_dot(d) and
                       strip_views(d->inputs().at(1)) == strip_views(step.state_in) and
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

// The per-chunk operands of one family, absent for chunk 0 of a zero start
using chunk_family = std::vector<std::optional<instruction_ref>>;

// f over the present entries; absent ones stay absent
template <class T, class F>
chunk_family gather(const std::vector<std::optional<T>>& xs, F f)
{
    chunk_family result(xs.size());
    std::transform(xs.begin(), xs.end(), result.begin(), [&](const auto& x) {
        return x.has_value() ? std::optional<instruction_ref>{f(*x)}
                             : std::optional<instruction_ref>{};
    });
    return result;
}

// A family as a view of its base: every chunk is the same view, each a
// constant number of elements further in
struct family
{
    instruction_ref base;
    shape view;
    std::size_t offset = 0;
    std::size_t delta  = 0;
};

std::optional<family> make_family(const chunk_family& chunks)
{
    std::optional<family> f;
    std::optional<std::size_t> first; // index of the first present chunk
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        if(not chunks[i].has_value())
            continue;
        auto o = origin_of(*chunks[i]);
        if(not f.has_value())
        {
            f     = family{o.base, o.view, o.offset, 0};
            first = i;
            continue;
        }
        if(o.base != f->base or o.view != f->view or o.offset < f->offset)
            return std::nullopt;
        const auto span = i - *first;
        const auto dist = o.offset - f->offset;
        if(dist % span != 0)
            return std::nullopt;
        const auto delta = dist / span;
        if(span > 1 and delta != f->delta)
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

// The output read of a state: the dot(q_i, S) that is not the read feeding
// the chunk's transition
std::optional<instruction_ref> output_read(instruction_ref state,
                                           const std::optional<chunk_step>& step)
{
    auto readers = dot_readers(state);
    if(step.has_value() and step->ws.has_value())
        erase_if(readers, [&](auto out) { return strip_views(*step->ws) == out; });
    if(readers.size() != 1)
        return std::nullopt;
    return readers.front();
}

// Follow the read through the views the simplifier leaves to its out
// pointwise and the concat
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

// The matched loop over n chunks: the state entering each chunk, its
// transition and its output, and the concat assembling the outputs. The
// entries of chunk 0 are absent when the walk starts from zero.
struct chunk_loop
{
    std::size_t chunks = 0;
    bool has_state     = false;
    instruction_ref final_state;
    instruction_ref concat;
    std::optional<instruction_ref> zero_out; // chunk 0's output view of a zero start
    std::vector<std::optional<instruction_ref>> states;
    std::vector<std::optional<chunk_step>> steps;
    std::vector<std::optional<chunk_output>> outputs;

    // the first chunk with a transition
    std::size_t first() const { return has_state ? 0 : 1; }
    const chunk_step& head() const { return *steps.at(first()); }
    const chunk_output& head_output() const { return *outputs.at(first()); }
    bool has_out() const
    {
        return std::any_of(outputs.begin(), outputs.end(), [](const auto& o) {
            return o.has_value() and o->out.has_value();
        });
    }
};

// The transitions of a module by the state they consume, and the pointwises
// that are transitions
struct step_map
{
    std::unordered_map<instruction_ref, chunk_step> by_state;
    std::unordered_set<instruction_ref> transitions;
};

// The operands of chunk 0 of a zero start come from the state entering
// chunk 1, which is its update: dot(kt_0, u0_0) or the precomputed view
std::vector<instruction_ref> zero_start_operands(instruction_ref state1)
{
    if(is_dot(state1))
        return state1->inputs();
    return {state1};
}

// Whether the chain starts from a zero state: its first state has the form
// of a chunk's update and completes the chunk operand families
bool starts_from_zero(const std::vector<chunk_step>& chain)
{
    const auto& head = chain.front();
    const auto ops   = zero_start_operands(head.state_in);
    if(head.kt.has_value())
    {
        if(ops.size() != 2)
            return false;
        chunk_family kts{ops[0]};
        chunk_family u0s{ops[1]};
        std::transform(chain.begin(), chain.end(), std::back_inserter(kts), [](const auto& s) {
            return s.kt;
        });
        std::transform(chain.begin(), chain.end(), std::back_inserter(u0s), [](const auto& s) {
            return s.u0;
        });
        return make_family(kts).has_value() and make_family(u0s).has_value();
    }
    if(not is_view(head.state_in))
        return false;
    chunk_family upds{head.state_in};
    std::transform(chain.begin(), chain.end(), std::back_inserter(upds), [](const auto& s) {
        return std::optional<instruction_ref>{s.upd};
    });
    return make_family(upds).has_value();
}

// The chain of transitions from a state, bounded by the concat the chunk
// outputs join: a transition whose chunk joins another concat starts the
// next recurrence
std::optional<chunk_loop> trace_loop(instruction_ref state, const step_map& steps)
{
    std::vector<chunk_step> chain;
    std::vector<std::optional<chunk_output>> outputs;
    std::optional<instruction_ref> concat;
    auto it = steps.by_state.find(state);
    assert(it != steps.by_state.end());
    while(it != steps.by_state.end())
    {
        auto out = trace_output(it->first, it->second);
        auto c   = out.has_value() ? std::optional<instruction_ref>{out->concat}
                                   : std::optional<instruction_ref>{};
        if(chain.empty())
            concat = c;
        else if(c != concat)
            break;
        chain.push_back(it->second);
        outputs.push_back(out);
        it = steps.by_state.find(it->second.state_out);
    }
    const auto& head = chain.front();
    if(std::any_of(chain.begin(), chain.end(), [&](const chunk_step& step) {
           return step.kt.has_value() != head.kt.has_value() or
                  step.read.has_value() != head.read.has_value() or
                  step.gamma.has_value() != head.gamma.has_value();
       }))
        return std::nullopt;

    chunk_loop loop;
    loop.has_state   = not starts_from_zero(chain);
    loop.chunks      = chain.size() + loop.first();
    loop.final_state = chain.back().state_out;
    loop.states.resize(loop.chunks);
    loop.steps.resize(loop.chunks);
    loop.outputs.resize(loop.chunks);
    std::copy(chain.begin(), chain.end(), loop.steps.begin() + loop.first());
    std::copy(outputs.begin(), outputs.end(), loop.outputs.begin() + loop.first());
    std::transform(chain.begin(),
                   chain.end(),
                   loop.states.begin() + loop.first(),
                   [](const chunk_step& s) { return s.state_in; });
    return loop;
}

// Every chunk's output joins the same concat, in chunk order, all through an
// out pointwise or none; chunk 0 of a zero start joins its pu0 view directly
bool check_concat(chunk_loop& loop)
{
    if(not loop.outputs.at(loop.first()).has_value())
        return false;
    const auto& head = loop.head_output();
    if(std::any_of(loop.outputs.begin(), loop.outputs.end(), [&](const auto& o) {
           return o.has_value() and
                  (o->concat != head.concat or o->out.has_value() != head.out.has_value());
       }))
        return false;
    loop.concat          = head.concat;
    const auto& joined   = loop.concat->inputs();
    const bool per_chunk = head.out.has_value();
    if(joined.size() != loop.chunks)
        return false;
    for(std::size_t i = 0; i < loop.chunks; i++)
    {
        const auto& o = loop.outputs[i];
        if(o.has_value())
        {
            if(strip_views(joined[i]) != o->out.value_or(o->read))
                return false;
        }
        else if(per_chunk and is_view(joined[i]))
        {
            loop.zero_out = joined[i];
        }
        else
        {
            return false;
        }
    }
    return true;
}

// The state accumulates in float unless the operands already are float or
// double; the kernel's accumulation type follows the same rule
shape::type_t accumulation_type(shape::type_t t)
{
    return (t == shape::float_type or t == shape::double_type) ? t : shape::float_type;
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
    // literals keep their value but join the body in its type
    for(auto ins : iterator_for(pm))
    {
        if(ins->name() != "@literal" or ins->get_shape().type() == type)
            continue;
        map[ins] = m.add_instruction(make_op("convert", {{"target_type", type}}),
                                     m.add_literal(ins->get_literal()));
    }
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
    assert(std::all_of(
        roles.begin(), roles.end(), [&](auto role) { return contains(pw->inputs(), role); }));
    auto body     = canonical_body(*pw->module_inputs().front(), type);
    module_ref rm = mpm.create_module(name);
    rm->set_bypass();
    std::unordered_map<instruction_ref, instruction_ref> map;
    for(std::size_t k = 0; k < roles.size(); k++)
        map[roles[k]] = rm->add_parameter(param_name(k), shape{type});
    rm->add_return(rm->fuse(body, pw->inputs(), &map));
    return rm;
}

bool same_modules(const chunk_family& pws, shape::type_t type)
{
    std::optional<module> first;
    for(const auto& pw : pws)
    {
        if(not pw.has_value())
            continue;
        auto body = canonical_body(*(*pw)->module_inputs().front(), type);
        if(not first.has_value())
            first = std::move(body);
        else if(not(*first == body))
            return false;
    }
    return true;
}

// Every chunk's step, read and out body must be the same pointwise
bool same_bodies(const chunk_loop& loop, shape::type_t type)
{
    const auto& head = loop.head();
    if(not same_modules(gather(loop.steps, [](const chunk_step& s) { return s.state_out; }), type))
        return false;
    if(head.read.has_value() and
       not same_modules(gather(loop.steps, [](const chunk_step& s) { return *s.read; }), type))
        return false;
    return not loop.has_out() or
           same_modules(gather(loop.outputs, [](const chunk_output& o) { return o.out; }), type);
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
    std::vector<int64_t> axes;
    const auto& lens = view.lens();
    // the matrix axes stay; unit axes before them go, last first
    for(auto axis = lens.size() - 2; axis > 0 and lens.size() - axes.size() > ndim; axis--)
    {
        if(lens[axis - 1] == 1)
            axes.push_back(axis - 1);
    }
    if(lens.size() - axes.size() != ndim)
        return std::nullopt;
    // squeeze with no axes would drop every unit axis
    if(axes.empty())
        return view;
    return make_op("squeeze", {{"axes", axes}}).compute_shape({view});
}

// Whether every instruction in inss comes before pos in the module
bool all_before(const module& m, const std::vector<instruction_ref>& inss, instruction_ref pos)
{
    std::unordered_set<instruction_ref> pending(inss.begin(), inss.end());
    for(auto ins : iterator_for(m))
    {
        if(ins == pos)
            break;
        pending.erase(ins);
    }
    return pending.empty();
}

// The op's shapes: the kernel writes (lead..., n * rows, dv); the concat's
// own layout is a reshape of that
chunk_scan scan_op(const chunk_loop& loop)
{
    chunk_scan op;
    op.chunks                        = loop.chunks;
    op.has_state                     = loop.has_state;
    op.state_shape                   = loop.final_state->get_shape();
    auto out_lens                    = op.state_shape.lens();
    const auto& q_lens               = loop.head_output().read->get_shape().lens();
    out_lens.at(out_lens.size() - 2) = q_lens.at(q_lens.size() - 2) * loop.chunks;
    op.output_shape                  = shape{loop.concat->get_shape().type(), out_lens};
    return op;
}

// The per-chunk operands of each family in the op's input order
std::vector<std::pair<std::string, chunk_family>> operand_families(const chunk_loop& loop)
{
    const auto& head = loop.head();
    std::vector<std::pair<std::string, chunk_family>> families;
    if(head.read.has_value())
        families.emplace_back("w", gather(loop.steps, [](const chunk_step& s) { return *s.w; }));
    families.emplace_back(
        "q", gather(loop.outputs, [](const chunk_output& o) { return o.read->inputs().at(0); }));
    if(head.kt.has_value())
    {
        auto kts = gather(loop.steps, [](const chunk_step& s) { return *s.kt; });
        auto u0s = gather(loop.steps, [](const chunk_step& s) { return *s.u0; });
        if(not loop.has_state)
        {
            const auto ops = zero_start_operands(*loop.states[1]);
            kts[0]         = ops.at(0);
            u0s[0]         = ops.at(1);
        }
        families.emplace_back("kt", kts);
        families.emplace_back("u0", u0s);
    }
    else
    {
        auto upds = gather(loop.steps, [](const chunk_step& s) { return s.upd; });
        if(not loop.has_state)
            upds[0] = zero_start_operands(*loop.states[1]).at(0);
        families.emplace_back("upd", upds);
    }
    if(head.gamma.has_value())
        families.emplace_back("gamma",
                              gather(loop.steps, [](const chunk_step& s) { return *s.gamma; }));
    if(loop.has_out())
    {
        auto pu0s = gather(loop.outputs, [](const chunk_output& o) { return o.pu0; });
        if(not loop.has_state)
            pu0s[0] = loop.zero_out;
        families.emplace_back("pu0", pu0s);
    }
    return families;
}

// Encode a family as a view of its base in the op and return the base; the
// leading axes are the problem axes, shared with the state
std::optional<instruction_ref>
add_family(chunk_scan& op, const std::string& name, const chunk_family& chunks)
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
    op.families.push_back(name);
    op.views.push_back(*view);
    op.offsets.push_back(f->offset);
    op.deltas.push_back(f->delta);
    return f->base;
}

// The step, read and out submodules in the kernel's order
std::vector<module_ref> scan_modules(module_pass_manager& mpm,
                                     const chunk_loop& loop,
                                     const std::string& prefix,
                                     shape::type_t type)
{
    const auto& head = loop.head();
    std::vector<module_ref> mods;
    mods.push_back(copy_pointwise(mpm, prefix + ":step", head.state_out, step_roles(head), type));
    if(head.read.has_value())
        mods.push_back(
            copy_pointwise(mpm, prefix + ":read", *head.read, {*head.u0, *head.ws}, type));
    if(loop.has_out())
    {
        const auto& o = loop.head_output();
        mods.push_back(copy_pointwise(mpm, prefix + ":out", *o.out, {*o.pu0, *o.qs_in}, type));
    }
    return mods;
}

bool fuse_loop(module_pass_manager& mpm, const chunk_loop& loop, const std::string& prefix)
{
    auto& m                  = mpm.get_module();
    auto op                  = scan_op(loop);
    const auto& concat_shape = loop.concat->get_shape();
    // the chunks are joined along the rows of the (.., n * rows, dv) output
    if(not concat_shape.standard() or concat_shape.elements() != op.output_shape.elements())
        return false;
    std::vector<instruction_ref> inputs;
    if(op.has_state)
        inputs.push_back(*loop.states[0]);
    std::optional<instruction_ref> q;
    for(const auto& [name, chunks] : operand_families(loop))
    {
        auto base = add_family(op, name, chunks);
        if(not base.has_value())
            return false;
        if(name == "q")
            q = base;
        inputs.push_back(*base);
    }
    // every input must be computed before the loop is replaced
    if(not all_before(m, inputs, loop.concat))
        return false;
    // the operands' type, as the kernel takes it from q
    const auto type = accumulation_type(q.value()->get_shape().type());
    if(not same_bodies(loop, type))
        return false;

    auto mods  = scan_modules(mpm, loop, prefix, type);
    auto scan  = m.insert_instruction(loop.concat, op, inputs, mods);
    auto out   = m.insert_instruction(loop.concat, make_op("get_tuple_elem", {{"index", 0}}), scan);
    auto state = m.insert_instruction(loop.concat, make_op("get_tuple_elem", {{"index", 1}}), scan);
    if(concat_shape.lens() != op.output_shape.lens())
        out = m.insert_instruction(
            loop.concat, make_op("reshape", {{"dims", concat_shape.lens()}}), out);
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
        if(not step.has_value())
            continue;
        steps.by_state.emplace(step->state_in, *step);
        steps.transitions.insert(ins);
    }
    // chains start at the states no transition produces; walk them in module
    // order so the fusions are deterministic
    std::deque<instruction_ref> heads;
    auto im = iterator_for(m);
    std::copy_if(im.begin(), im.end(), std::back_inserter(heads), [&](auto ins) {
        return steps.by_state.count(ins) > 0 and steps.transitions.count(ins) == 0;
    });
    std::vector<chunk_loop> loops;
    while(not heads.empty())
    {
        auto loop = trace_loop(heads.front(), steps);
        heads.pop_front();
        if(not loop.has_value())
            continue;
        // a recurrence fed by this one's final state is traced next
        if(steps.by_state.count(loop->final_state) > 0)
            heads.push_back(loop->final_state);
        if(check_concat(*loop))
            loops.push_back(std::move(*loop));
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
