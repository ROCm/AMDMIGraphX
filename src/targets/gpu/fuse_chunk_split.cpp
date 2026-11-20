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
#include <migraphx/ranges.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/enum.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <cassert>
#include <deque>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Walks the chunks of a linear recurrence with the state S resident. For
// chunk i, with u the input of its update and o_i its rows of the output:
//   u   = read(update_input_i, read_matrix_i S) or update_input_i
//   o_i = out(output_base_i, output_matrix_i S) or output_matrix_i S
//   S   = step(S, decay_i, update_matrix_i u or state_update_i)
// Each per-chunk operand family is one input holding its chunks stacked ahead
// of the matrix axes, (lead..., chunks, m, n), with lead the problem axes of
// the state; `families` names the present ones in the order of the enum,
// which is the op's input order after the initial state when there is one. A
// recurrence has either update_matrix and update_input or state_update.
// The submodules are the step, read and out pointwise bodies in that order,
// each with its inputs in the order listed above.
struct chunk_scan
{
    MIGRAPHX_NESTED_ENUM_CLASS(family,
                               read_matrix,
                               output_matrix,
                               update_matrix,
                               update_input,
                               state_update,
                               decay,
                               output_base)

    shape output_shape;
    shape state_shape;
    std::vector<family> families;
    std::size_t chunks = 0;
    bool has_state     = false; // S_0 input, else the walk starts from zero

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"),
                    f(self.state_shape, "state_shape"),
                    f(self.families, "families"),
                    f(self.chunks, "chunks"),
                    f(self.has_state, "has_state"));
    }

    std::string name() const { return "gpu::chunk_scan"; }

    bool has(family role) const { return contains(families, role); }

    // The lens of a family's input for chunks of `rows` rows. A zero start
    // never reads the state for chunk 0, so the read and output matrices
    // hold one chunk less.
    std::vector<std::size_t> family_lens(family role, std::size_t rows) const
    {
        const auto& slens = state_shape.lens();
        const auto dk     = slens[slens.size() - 2];
        const auto dv     = slens.back();
        const bool reads  = role == family::read_matrix or role == family::output_matrix;
        const bool tail   = reads and not has_state;
        std::vector<std::size_t> lens(slens.begin(), slens.end() - 2);
        lens.push_back(tail ? chunks - 1 : chunks);
        switch(role)
        {
        case family::read_matrix:
        case family::output_matrix: lens.insert(lens.end(), {rows, dk}); break;
        case family::update_matrix: lens.insert(lens.end(), {dk, rows}); break;
        case family::update_input:
        case family::output_base: lens.insert(lens.end(), {rows, dv}); break;
        case family::state_update:
        case family::decay: lens.insert(lens.end(), {dk, dv}); break;
        }
        return lens;
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        const std::size_t state_inputs = has_state ? 1 : 0;
        check_shapes{inputs, *this}.has(families.size() + state_inputs);
        if(chunks == 0 or state_shape.ndim() < 2)
            MIGRAPHX_THROW("chunk_scan: needs at least one chunk and a matrix state");
        if(mods.size() !=
           1 + (has(family::read_matrix) ? 1 : 0) + (has(family::output_base) ? 1 : 0))
            MIGRAPHX_THROW("chunk_scan: wrong number of submodules");
        if(has_state and inputs.front().lens() != state_shape.lens())
            MIGRAPHX_THROW("chunk_scan: the initial state must have the lens of the state");
        auto output_matrix = std::find(families.begin(), families.end(), family::output_matrix);
        if(output_matrix == families.end())
            MIGRAPHX_THROW("chunk_scan: needs the output_matrix family");
        const auto& matrix_lens = inputs[state_inputs + (output_matrix - families.begin())].lens();
        if(matrix_lens.size() != state_shape.ndim() + 1)
            MIGRAPHX_THROW("chunk_scan: family output_matrix has the wrong rank");
        const auto rows = matrix_lens[matrix_lens.size() - 2];
        auto bad        = std::mismatch(families.begin(),
                                        families.end(),
                                        inputs.begin() + state_inputs,
                                        [&](family role, const shape& input) {
                                     return input.lens() == family_lens(role, rows);
                                        });
        if(bad.first != families.end())
            MIGRAPHX_THROW("chunk_scan: family " + to_string(*bad.first) + " has lens {" +
                           to_string_range(bad.second->lens()) + "} instead of {" +
                           to_string_range(family_lens(*bad.first, rows)) + "}");
        auto out_lens                    = state_shape.lens();
        out_lens.at(out_lens.size() - 2) = rows * chunks;
        if(output_shape.lens() != out_lens)
            MIGRAPHX_THROW("chunk_scan: the output must stack the chunks along its rows");
        return shape{{output_shape, state_shape}};
    }
};
MIGRAPHX_REGISTER_OP(chunk_scan);

namespace {

// The views a shape_transform_descriptor describes; a slice is followed
// separately, since it selects elements
bool is_view(instruction_ref ins)
{
    static const std::unordered_set<std::string> names = {
        "reshape", "squeeze", "unsqueeze", "flatten", "transpose", "multibroadcast", "broadcast"};
    return contains(names, ins->name()) and ins->inputs().size() == 1;
}

bool is_slice(instruction_ref ins) { return ins->name() == "slice" and ins->inputs().size() == 1; }

bool is_dot(instruction_ref ins) { return ins->name() == "dot"; }

// The instruction a chain of views reads, with the views in the order they
// are applied
std::pair<instruction_ref, std::vector<operation>> view_chain(instruction_ref ins)
{
    return get_input_ops_if(ins, &is_view);
}

instruction_ref view_root(instruction_ref ins) { return view_chain(ins).first; }

// Whether ins repeats elements of what it reads, through its views and slices
bool is_broadcast_view(instruction_ref ins)
{
    auto [root, ops] = view_chain(ins);
    if(shape_transform_descriptor::create(root->get_shape().lens(), ops).has_broadcast())
        return true;
    return is_slice(root) and is_broadcast_view(root->inputs().front());
}

// The views that take src to ins: ins reads src, or both are views of the
// same instruction, in which case the views of src are undone first
std::optional<std::vector<operation>> views_between(instruction_ref src, instruction_ref ins)
{
    auto [root, ops] =
        get_input_ops_if(ins, [&](instruction_ref x) { return x != src and is_view(x); });
    if(root == src)
        return ops;
    auto [base, src_ops] = view_chain(src);
    if(root != base)
        return std::nullopt;
    auto undo = shape_transform_descriptor::create(base->get_shape().lens(), src_ops).invert();
    if(undo.empty())
        return std::nullopt;
    auto result = undo.generate();
    result.insert(result.end(), ops.begin(), ops.end());
    return result;
}

// Whether ins holds the elements of src in the same order: the views between
// them only regroup axes, without selecting, repeating or reordering elements
bool same_order(instruction_ref src, instruction_ref ins)
{
    auto ops = views_between(src, ins);
    if(not ops.has_value())
        return false;
    auto desc = shape_transform_descriptor::create(src->get_shape().lens(), *ops);
    return not desc.empty() and not desc.has_broadcast() and not desc.is_transposed();
}

// Whether ins holds the matrices of src: the same elements in the same order
// under the same matrix axes. The scan keeps the state and its products as
// (rows, columns) tiles, so nothing else can be read from them.
bool same_matrix(instruction_ref src, instruction_ref ins)
{
    const auto& x = src->get_shape().lens();
    const auto& y = ins->get_shape().lens();
    return x.size() >= 2 and y.size() >= 2 and std::equal(x.end() - 2, x.end(), y.end() - 2) and
           same_order(src, ins);
}

// The instruction a chain of single-use views feeds
instruction_ref follow_views(instruction_ref ins)
{
    while(ins->outputs().size() == 1 and is_view(ins->outputs().front()))
        ins = ins->outputs().front();
    return ins;
}

// The dots reading the matrices of src as their right operand, among those
// reading ins or a view of it
std::vector<instruction_ref> dot_readers(instruction_ref src, instruction_ref ins)
{
    std::vector<instruction_ref> readers;
    for(auto out : ins->outputs())
    {
        if(is_dot(out) and out->inputs().at(1) == ins)
        {
            if(same_matrix(src, ins))
                readers.push_back(out);
        }
        else if(is_view(out))
        {
            auto through = dot_readers(src, out);
            readers.insert(readers.end(), through.begin(), through.end());
        }
    }
    return readers;
}

// A view of what ins is a view of may hold its matrices too, so the walk
// starts from there
std::vector<instruction_ref> dot_readers(instruction_ref ins)
{
    return dot_readers(ins, view_root(ins));
}

// Whether a dot reads the matrices of ins as its right operand: the state
// is, and the decay and precomputed updates are not
bool read_by_dot(instruction_ref ins) { return not dot_readers(ins).empty(); }

// One transition S' = step(S, [decay_i], state_update_i) with the
// instructions it consumes from the chunk's operand families
struct chunk_step
{
    instruction_ref state_in;
    instruction_ref state_out;
    instruction_ref state_update; // the precomputed view, or the dot(update_matrix, u)
    std::optional<instruction_ref> decay;
    std::optional<instruction_ref> update_matrix;
    std::optional<instruction_ref> update_input;
    std::optional<instruction_ref> read_matrix;
    std::optional<instruction_ref> read; // u = read(update_input, dot(read_matrix, S))
    // the read's input that comes from dot(read_matrix, S)
    std::optional<instruction_ref> read_product;
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
           return in->outputs().size() != 1 or not(is_dot(in) or is_view(in) or is_slice(in));
       }))
        return std::nullopt;
    auto update_it = std::find_if(rest.begin(), rest.end(), is_dot);
    if(update_it == rest.end())
        update_it = std::find_if_not(rest.begin(), rest.end(), &is_broadcast_view);
    if(update_it == rest.end())
        return std::nullopt;
    step.state_update = *update_it;
    rest.erase(update_it);
    if(rest.size() > 1)
        return std::nullopt;
    if(not rest.empty())
    {
        if(not is_broadcast_view(rest.front()))
            return std::nullopt;
        step.decay = rest.front();
    }
    if(is_dot(step.state_update))
    {
        step.update_matrix  = step.state_update->inputs().at(0);
        auto update_operand = step.state_update->inputs().at(1);
        auto u              = view_root(update_operand);
        if(u->name() == "pointwise" and u->inputs().size() == 2)
        {
            // u = read(update_input, dot(read_matrix, S)); the views on the
            // way keep the elements in order, and the dot reads the matrices
            // of the state
            auto product = std::find_if(u->inputs().begin(), u->inputs().end(), [&](auto in) {
                auto d = view_root(in);
                return is_dot(d) and d->outputs().size() == 1 and same_order(d, in) and
                       same_matrix(step.state_in, d->inputs().at(1));
            });
            if(product == u->inputs().end() or not same_order(u, update_operand))
                return std::nullopt;
            step.read         = u;
            step.read_product = *product;
            step.read_matrix  = view_root(*product)->inputs().at(0);
            step.update_input = u->inputs().at(*product == u->inputs().front() ? 1 : 0);
        }
        else
        {
            step.update_input = update_operand;
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

// The range a slice selects along each of its axes
struct slice_range
{
    std::vector<std::size_t> axes;
    std::vector<std::size_t> starts;
    std::vector<std::size_t> ends;

    static slice_range of(instruction_ref slice)
    {
        const auto v = slice->normalized_operator().to_value();
        return {v.at("axes").to_vector<std::size_t>(),
                v.at("starts").to_vector<std::size_t>(),
                v.at("ends").to_vector<std::size_t>()};
    }

    // The same axes with axis k advanced by i times its length
    slice_range advance(std::size_t k, std::size_t i) const
    {
        auto result      = *this;
        const auto len   = ends[k] - starts[k];
        result.starts[k] = starts[k] + i * len;
        result.ends[k]   = result.starts[k] + len;
        return result;
    }

    // The slice of an input of lens `lens`, or nothing when it takes all of it
    std::optional<operation> to_op(const std::vector<std::size_t>& lens) const
    {
        auto ks = range(axes.size());
        if(std::all_of(ks.begin(), ks.end(), [&](auto k) {
               return starts[k] == 0 and ends[k] == lens.at(axes[k]);
           }))
            return std::nullopt;
        return make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}});
    }

    friend bool operator==(const slice_range& x, const slice_range& y)
    {
        return std::tie(x.axes, x.starts, x.ends) == std::tie(y.axes, y.starts, y.ends);
    }
};

// A transpose of a rank `rank` tensor that moves axis `from` to `to`
operation move_axis(std::size_t rank, std::size_t from, std::size_t to)
{
    std::vector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + from);
    perm.insert(perm.begin() + to, from);
    return make_op("transpose", {{"permutation", perm}});
}

// The chunks of a family stacked along a new leading axis, as views of one
// instruction: (n, lens of a chunk...)
struct chunk_stack
{
    instruction_ref base;
    std::optional<operation> slice; // the part of base the chunks cover
    std::vector<operation> ops;     // from that part to the stack
    std::vector<std::size_t> lens;  // of the part
    std::size_t n = 0;
};

// The views of a chunk over the instruction they read
shape_transform_descriptor chunk_views(instruction_ref chunk)
{
    auto [root, ops] = view_chain(chunk);
    return shape_transform_descriptor::create(root->get_shape().lens(), ops);
}

// One chunk is its own stack
chunk_stack stack_chunk(instruction_ref chunk)
{
    auto [root, ops] = view_chain(chunk);
    ops.push_back(make_op("unsqueeze", {{"axes", {0}}}));
    return {root, std::nullopt, ops, root->get_shape().lens(), 1};
}

// Several chunks stack when they are consecutive, equal slices of one
// instruction along one axis, each under the same views: the sliced axis
// splits into (n, len) and the views apply to every chunk
std::optional<chunk_stack> stack_chunks(const std::vector<instruction_ref>& chunks)
{
    assert(not chunks.empty());
    if(chunks.size() == 1)
        return stack_chunk(chunks.front());
    std::vector<instruction_ref> slices;
    std::transform(chunks.begin(), chunks.end(), std::back_inserter(slices), &view_root);
    if(not std::all_of(slices.begin(), slices.end(), &is_slice))
        return std::nullopt;
    auto base = slices.front()->inputs().front();
    auto desc = chunk_views(chunks.front());
    if(desc.empty() or
       not std::all_of(
           chunks.begin(), chunks.end(), [&](auto c) { return chunk_views(c) == desc; }) or
       not std::all_of(
           slices.begin(), slices.end(), [&](auto s) { return s->inputs().front() == base; }))
        return std::nullopt;

    const auto first = slice_range::of(slices.front());
    const auto next  = slice_range::of(slices.at(1));
    if(next.axes != first.axes)
        return std::nullopt;
    // the axis the chunks advance along
    auto advanced =
        std::mismatch(first.starts.begin(), first.starts.end(), next.starts.begin()).first;
    if(advanced == first.starts.end())
        return std::nullopt;
    const std::size_t k = std::distance(first.starts.begin(), advanced);
    if(first.ends[k] <= first.starts[k])
        return std::nullopt;
    const auto n = chunks.size();
    auto is      = range(n);
    if(not std::all_of(is.begin(), is.end(), [&](auto i) {
           return slice_range::of(slices[i]) == first.advance(k, i);
       }))
        return std::nullopt;

    const auto len = first.ends[k] - first.starts[k];
    auto cover     = first;
    cover.ends[k]  = first.starts[k] + n * len;
    chunk_stack result;
    result.base       = base;
    result.slice      = cover.to_op(base->get_shape().lens());
    result.n          = n;
    result.lens       = slices.front()->get_shape().lens();
    const auto axis   = first.axes[k];
    result.lens[axis] = n * len;
    auto split_lens   = result.lens;
    split_lens[axis]  = len;
    split_lens.insert(split_lens.begin() + axis, n);
    result.ops.push_back(make_op("reshape", {{"dims", split_lens}}));
    if(axis != 0)
        result.ops.push_back(move_axis(split_lens.size(), axis, 0));
    desc.prepend_axis(n);
    auto chain = desc.generate();
    result.ops.insert(result.ops.end(), chain.begin(), chain.end());
    return result;
}

// The present chunks of a family: all of them, or all but chunk 0
std::optional<std::vector<instruction_ref>> present_chunks(const chunk_family& chunks)
{
    auto first =
        std::find_if(chunks.begin(), chunks.end(), [](const auto& c) { return c.has_value(); });
    if(first == chunks.end() or
       std::any_of(first, chunks.end(), [](const auto& c) { return not c.has_value(); }))
        return std::nullopt;
    std::vector<instruction_ref> result;
    std::transform(
        first, chunks.end(), std::back_inserter(result), [](const auto& c) { return *c; });
    return result;
}

bool is_family(const chunk_family& chunks)
{
    auto present = present_chunks(chunks);
    return present.has_value() and stack_chunks(*present).has_value();
}

// The output of one chunk: the read output_matrix_i S of its state, the out
// pointwise it may pass through, and the concat joining it
struct chunk_output
{
    instruction_ref read;
    std::optional<instruction_ref> out;
    // the out's input that comes from the read
    std::optional<instruction_ref> output_product;
    std::optional<instruction_ref> output_base;
    instruction_ref concat;
};

// The output read of a state: the dot(output_matrix_i, S) that is not the
// read feeding the chunk's transition
std::optional<instruction_ref> output_read(instruction_ref state,
                                           const std::optional<chunk_step>& step)
{
    auto readers = dot_readers(state);
    if(step.has_value() and step->read_product.has_value())
        erase_if(readers, [&](auto out) { return view_root(*step->read_product) == out; });
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
        bool first      = same_order(*read, ins.front());
        if(not first and not same_order(*read, ins.back()))
            return std::nullopt;
        result.out            = user;
        result.output_product = ins.at(first ? 0 : 1);
        result.output_base    = ins.at(first ? 1 : 0);
        user                  = follow_views(user);
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
// chunk 1, which is its update: dot(update_matrix_0, update_input_0) or the
// precomputed view
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
    if(head.update_matrix.has_value())
    {
        if(ops.size() != 2)
            return false;
        chunk_family update_matrices{ops[0]};
        chunk_family update_inputs{ops[1]};
        std::transform(chain.begin(),
                       chain.end(),
                       std::back_inserter(update_matrices),
                       [](const auto& s) { return s.update_matrix; });
        std::transform(chain.begin(),
                       chain.end(),
                       std::back_inserter(update_inputs),
                       [](const auto& s) { return s.update_input; });
        return is_family(update_matrices) and is_family(update_inputs);
    }
    chunk_family state_updates{head.state_in};
    std::transform(chain.begin(),
                   chain.end(),
                   std::back_inserter(state_updates),
                   [](const auto& s) { return std::optional<instruction_ref>{s.state_update}; });
    return is_family(state_updates);
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
           return step.update_matrix.has_value() != head.update_matrix.has_value() or
                  step.read.has_value() != head.read.has_value() or
                  step.decay.has_value() != head.decay.has_value();
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
// out pointwise or none; chunk 0 of a zero start joins its output_base view
// directly
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
            if(not same_order(o->out.value_or(o->read), joined[i]))
                return false;
        }
        else if(per_chunk and (is_view(joined[i]) or is_slice(joined[i])))
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

// A chunk's step, read or out pointwise with its inputs in the order the
// kernel passes them
struct scan_body
{
    instruction_ref pw;
    std::vector<instruction_ref> roles;

    // The canonical body with its parameters in the order of the roles, so
    // bodies that take their inputs in different orders compare by what they
    // compute
    module make(shape::type_t type) const
    {
        assert(std::all_of(
            roles.begin(), roles.end(), [&](auto role) { return contains(pw->inputs(), role); }));
        auto body = canonical_body(*pw->module_inputs().front(), type);
        module m;
        std::unordered_map<instruction_ref, instruction_ref> map;
        for(std::size_t k = 0; k < roles.size(); k++)
            map[roles[k]] = m.add_parameter(param_name(k), shape{type});
        m.add_return(m.fuse(body, pw->inputs(), &map));
        return m;
    }
};

scan_body step_body(const chunk_step& step)
{
    scan_body result{step.state_out, {step.state_in}};
    if(step.decay.has_value())
        result.roles.push_back(*step.decay);
    result.roles.push_back(step.state_update);
    return result;
}

std::optional<scan_body> read_body(const chunk_step& step)
{
    if(not step.read.has_value())
        return std::nullopt;
    return scan_body{*step.read, {*step.update_input, *step.read_product}};
}

std::optional<scan_body> out_body(const chunk_output& output)
{
    if(not output.out.has_value())
        return std::nullopt;
    return scan_body{*output.out, {*output.output_base, *output.output_product}};
}

// Whether the chunks that have the body `f` yields all compute the same
template <class T, class F>
bool same_modules(const std::vector<std::optional<T>>& chunks, shape::type_t type, F f)
{
    std::vector<std::optional<scan_body>> bodies(chunks.size());
    std::transform(chunks.begin(), chunks.end(), bodies.begin(), [&](const auto& chunk) {
        return chunk.has_value() ? std::optional<scan_body>{f(*chunk)} : std::nullopt;
    });
    std::vector<module> mods;
    transform_if(
        bodies.begin(),
        bodies.end(),
        std::back_inserter(mods),
        [](const auto& body) { return body.has_value(); },
        [&](const auto& body) { return body->make(type); });
    return std::adjacent_find(mods.begin(), mods.end(), std::not_equal_to<>{}) == mods.end();
}

// Every chunk's step, read and out body must be the same pointwise
bool same_bodies(const chunk_loop& loop, shape::type_t type)
{
    return same_modules(loop.steps, type, &step_body) and
           same_modules(loop.steps, type, &read_body) and
           same_modules(loop.outputs, type, &out_body);
}

module_ref add_scan_module(module_pass_manager& mpm,
                           const std::string& name,
                           const scan_body& body,
                           shape::type_t type)
{
    module_ref rm = mpm.create_module(name, body.make(type));
    rm->set_bypass();
    return rm;
}

// Whether the concat joins whole chunks one after another: the axes from the
// concat axis on hold the rows and columns of one chunk, as the (lead...,
// n * rows, dv) output of the scan does
bool joins_chunks(instruction_ref concat, std::size_t chunk_elements)
{
    const auto axis  = concat->normalized_operator().to_value().at("axis").to<std::size_t>();
    const auto& lens = concat->inputs().front()->get_shape().lens();
    return std::accumulate(lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<>{}) ==
           chunk_elements;
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

// Whichever of a and b comes first in the module
instruction_ref first_of(const module& m, instruction_ref a, instruction_ref b)
{
    auto inss = iterator_for(m);
    auto it   = std::find_if(
        inss.begin(), inss.end(), [&](instruction_ref ins) { return ins == a or ins == b; });
    assert(it != inss.end());
    return *it;
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
    const auto& read_lens            = loop.head_output().read->get_shape().lens();
    out_lens.at(out_lens.size() - 2) = read_lens.at(read_lens.size() - 2) * loop.chunks;
    op.output_shape                  = shape{loop.concat->get_shape().type(), out_lens};
    return op;
}

// The per-chunk operands of a family; a dot reads its operand as matrices,
// a pointwise only element by element
struct operand_family
{
    chunk_scan::family role;
    chunk_family chunks;
    bool dot_operand = false;
};

// The families in the op's input order
std::vector<operand_family> operand_families(const chunk_loop& loop)
{
    using family     = chunk_scan::family;
    const auto& head = loop.head();
    std::vector<operand_family> families;
    if(head.read.has_value())
        families.push_back({family::read_matrix,
                            gather(loop.steps, [](const chunk_step& s) { return *s.read_matrix; }),
                            true});
    families.push_back(
        {family::output_matrix,
         gather(loop.outputs, [](const chunk_output& o) { return o.read->inputs().at(0); }),
         true});
    if(head.update_matrix.has_value())
    {
        auto update_matrices =
            gather(loop.steps, [](const chunk_step& s) { return *s.update_matrix; });
        auto update_inputs =
            gather(loop.steps, [](const chunk_step& s) { return *s.update_input; });
        if(not loop.has_state)
        {
            const auto ops     = zero_start_operands(*loop.states[1]);
            update_matrices[0] = ops.at(0);
            update_inputs[0]   = ops.at(1);
        }
        families.push_back({family::update_matrix, update_matrices, true});
        families.push_back({family::update_input, update_inputs, not head.read.has_value()});
    }
    else
    {
        auto state_updates = gather(loop.steps, [](const chunk_step& s) { return s.state_update; });
        if(not loop.has_state)
            state_updates[0] = zero_start_operands(*loop.states[1]).at(0);
        families.push_back({family::state_update, state_updates});
    }
    if(head.decay.has_value())
        families.push_back(
            {family::decay, gather(loop.steps, [](const chunk_step& s) { return *s.decay; })});
    if(loop.has_out())
    {
        auto output_bases =
            gather(loop.outputs, [](const chunk_output& o) { return o.output_base; });
        if(not loop.has_state)
            output_bases[0] = loop.zero_out;
        families.push_back({family::output_base, output_bases});
    }
    return families;
}

// One input of the scan: the views of base that stack a family's chunks
struct family_input
{
    instruction_ref base;
    std::vector<operation> ops;
};

// The views stacking a family's chunks into `lens`, the (lead..., n, m, k)
// its role takes. The stack axis goes ahead of the axes of a chunk that hold
// one matrix, which keeps the elements of every problem in order; a dot
// operand has the matrix as its last two axes.
std::optional<family_input> make_family_input(const operand_family& f,
                                              const std::vector<std::size_t>& lens)
{
    auto present = present_chunks(f.chunks);
    if(not present.has_value() or present->size() != lens[lens.size() - 3])
        return std::nullopt;
    auto stack = stack_chunks(*present);
    if(not stack.has_value())
        return std::nullopt;
    const auto& clens = present->front()->get_shape().lens();
    if(f.dot_operand and
       (clens.size() < 2 or not std::equal(clens.end() - 2, clens.end(), lens.end() - 2)))
        return std::nullopt;
    // the products of the last j lens of a chunk
    std::vector<std::size_t> tails(clens.size() + 1, 1);
    std::partial_sum(clens.rbegin(), clens.rend(), tails.begin() + 1, std::multiplies<>{});
    if(tails.back() * stack->n != elements(lens))
        return std::nullopt;
    auto matrix = std::find(tails.begin(), tails.end(), lens[lens.size() - 2] * lens.back());
    if(matrix == tails.end())
        return std::nullopt;
    const std::size_t axis = clens.size() - std::distance(tails.begin(), matrix);
    auto ops               = stack->ops;
    if(axis != 0)
        ops.push_back(move_axis(clens.size() + 1, 0, axis));
    ops.push_back(make_op("reshape", {{"dims", lens}}));
    ops = optimize_shape_transforms(stack->lens, ops);
    if(stack->slice.has_value())
        ops.insert(ops.begin(), *stack->slice);
    return family_input{stack->base, ops};
}

// The step, read and out submodules in the kernel's order
std::vector<module_ref> scan_modules(module_pass_manager& mpm,
                                     const chunk_loop& loop,
                                     const std::string& prefix,
                                     shape::type_t type)
{
    const auto& head = loop.head();
    std::vector<module_ref> mods{add_scan_module(mpm, prefix + ":step", step_body(head), type)};
    if(auto read = read_body(head))
        mods.push_back(add_scan_module(mpm, prefix + ":read", *read, type));
    if(auto out = out_body(loop.head_output()))
        mods.push_back(add_scan_module(mpm, prefix + ":out", *out, type));
    return mods;
}

bool fuse_loop(module_pass_manager& mpm, const chunk_loop& loop, const std::string& prefix)
{
    auto& m                  = mpm.get_module();
    auto op                  = scan_op(loop);
    const auto& concat_shape = loop.concat->get_shape();
    const auto& out_lens     = op.output_shape.lens();
    const auto rows          = out_lens[out_lens.size() - 2] / op.chunks;
    if(not joins_chunks(loop.concat, rows * out_lens.back()))
        return false;
    std::vector<family_input> families;
    for(const auto& f : operand_families(loop))
    {
        auto input = make_family_input(f, op.family_lens(f.role, rows));
        if(not input.has_value())
            return false;
        op.families.push_back(f.role);
        families.push_back(*input);
    }
    // The scan replaces the concat and the final state, so it goes ahead of
    // both for their users to follow it. Every input must come before that:
    // one that does not may be computed from the final state.
    const auto pos = first_of(m, loop.concat, loop.final_state);
    std::vector<instruction_ref> bases;
    if(op.has_state)
        bases.push_back(*loop.states[0]);
    std::transform(families.begin(), families.end(), std::back_inserter(bases), [](const auto& f) {
        return f.base;
    });
    if(not all_before(m, bases, pos))
        return false;
    // the operands' type, as the kernel takes it from the output matrix
    const auto q =
        std::find(op.families.begin(), op.families.end(), chunk_scan::family::output_matrix);
    const auto type = accumulation_type(families[q - op.families.begin()].base->get_shape().type());
    if(not same_bodies(loop, type))
        return false;

    std::vector<instruction_ref> inputs;
    if(op.has_state)
        inputs.push_back(*loop.states[0]);
    std::transform(families.begin(),
                   families.end(),
                   std::back_inserter(inputs),
                   [&](const auto& f) { return insert_ops(m, pos, f.ops, f.base); });
    auto mods  = scan_modules(mpm, loop, prefix, type);
    auto scan  = m.insert_instruction(pos, op, inputs, mods);
    auto out   = m.insert_instruction(pos, make_op("get_tuple_elem", {{"index", 0}}), scan);
    auto state = m.insert_instruction(pos, make_op("get_tuple_elem", {{"index", 1}}), scan);
    if(concat_shape.lens() != op.output_shape.lens())
        out = m.insert_instruction(pos, make_op("reshape", {{"dims", concat_shape.lens()}}), out);
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
