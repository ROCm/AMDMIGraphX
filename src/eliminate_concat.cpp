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
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/output_iterator.hpp>

#include <migraphx/dfor.hpp>
#include <migraphx/tune_axis.hpp>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

optional<std::pair<std::vector<operation>, shape>> invert_view_chain(
    const std::vector<operation>& chain, const shape& input_shape, const shape& view_shape)
{
    std::vector<operation> fwd;
    std::transform(chain.begin(), chain.end(), std::back_inserter(fwd), [](const operation& op) {
        if(op.name() == "reshape_lazy")
            return make_op("reshape", {{"dims", op.to_value().at("dims")}});
        return op;
    });
    auto inv = shape_transform_descriptor::create(input_shape.lens(), fwd).invert();
    if(inv.empty())
        return std::nullopt;
    auto result = inv.generate();
    // Reshapes need to be lazy so the view is aliased instead of copied at runtime
    std::transform(result.begin(), result.end(), result.begin(), [](const operation& op) {
        if(op.name() != "reshape")
            return op;
        return make_op("reshape_lazy", op.to_value());
    });
    try
    {
        auto s = std::accumulate(
            result.begin(), result.end(), view_shape, [](const shape& acc, const operation& op) {
                std::vector<shape> inputs = {acc};
                if(op.output_alias(inputs).empty())
                    MIGRAPHX_THROW("Not a view operator: " + op.name());
                auto next = op.compute_shape(inputs);
                if(next.elements() != acc.elements())
                    MIGRAPHX_THROW("Not an element-preserving view: " + op.name());
                return next;
            });
        if(s.lens() != input_shape.lens())
            return std::nullopt;
        return std::make_pair(std::move(result), s);
    }
    catch(const migraphx::exception&)
    {
        return std::nullopt;
    }
}

namespace {

const std::unordered_set<std::string>& invertible_view_names()
{
    static const std::unordered_set<std::string> names = {
        "reshape", "reshape_lazy", "squeeze", "unsqueeze", "flatten", "transpose"};
    return names;
}

// Walk up from a concat input through single-use invertible view ops to the
// instruction that actually writes the data. Returns the producer and the
// view ops in producer-to-input order.
std::pair<instruction_ref, std::vector<operation>> get_view_chain(instruction_ref input)
{
    std::vector<operation> ops;
    auto producer = input;
    while(contains(invertible_view_names(), producer->name()) and producer->inputs().size() == 1 and
          producer->outputs().size() == 1)
    {
        ops.push_back(producer->get_operator());
        producer = producer->inputs().front();
    }
    std::reverse(ops.begin(), ops.end());
    return {producer, ops};
}

bool is_packed(const shape& s, std::size_t axis)
{
    auto alens  = s.lens();
    alens[axis] = 1;
    return shape{s.type(), alens, s.strides()}.packed();
}

struct concat_optimizer
{
    module* m = nullptr;
    allocation_model am;
    const concat_optimization* opt = nullptr;

    static operation make_slice(instruction_ref input, std::size_t axis, std::size_t start)
    {
        return make_op("slice",
                       {{"axes", {axis}},
                        {"starts", {start}},
                        {"ends", {input->get_shape().lens()[axis] + start}}});
    }

    static instruction_ref get_output_alias(instruction_ref ins)
    {
        auto aliases = instruction::get_output_alias(ins, true);
        if(aliases.size() != 1)
            return ins;
        // cppcheck-suppress returnDanglingLifetime
        return aliases.front();
    }

    bool is_allocation(instruction_ref ins) const
    {
        return ins->name() == "allocate" or ins->name() == am.name();
    }

    // True if giving the instructions in `shapes` their new shapes would make a returned value
    // non-standard. Users recompute their shapes from their inputs, as instruction::replace
    // does, so this follows views as well as ops that keep the layout of their input, such as
    // convert. The users in `skip` are rewritten by the caller.
    bool makes_output_non_standard(std::unordered_map<instruction_ref, shape> shapes,
                                   const std::vector<instruction_ref>& skip) const
    {
        std::vector<instruction_ref> seeds;
        std::transform(shapes.begin(), shapes.end(), std::back_inserter(seeds), [](const auto& p) {
            return p.first;
        });
        auto stack = seeds;
        while(not stack.empty())
        {
            auto ins = stack.back();
            stack.pop_back();
            for(auto out : ins->outputs())
            {
                if(contains(skip, out) or contains(seeds, out) or out->name() == "@return")
                    continue;
                std::vector<shape> inputs;
                std::transform(out->inputs().begin(),
                               out->inputs().end(),
                               std::back_inserter(inputs),
                               [&](instruction_ref x) {
                                   auto it = shapes.find(x);
                                   return it == shapes.end() ? x->get_shape() : it->second;
                               });
                shape s;
                try
                {
                    const auto& mods = out->module_inputs();
                    s = mods.empty() ? out->get_operator().compute_shape(inputs)
                                     : out->get_operator().compute_shape(inputs, mods);
                }
                catch(const std::exception&)
                {
                    return true;
                }
                auto it = shapes.find(out);
                if(s == (it == shapes.end() ? out->get_shape() : it->second))
                    continue;
                shapes[out] = s;
                stack.push_back(out);
            }
        }
        auto returns = m->get_returns();
        return std::any_of(returns.begin(), returns.end(), [&](instruction_ref r) {
            auto it = shapes.find(r);
            return it != shapes.end() and it->second != r->get_shape() and
                   not it->second.standard();
        });
    }

    // Writing the producer into a view of the super buffer replaces its allocation for every
    // user, and both take the layout of the view
    bool moves_output_layout(instruction_ref concat,
                             instruction_ref producer,
                             const shape& view_shape) const
    {
        return makes_output_non_standard(
            {{get_output_alias(producer), view_shape}, {producer, view_shape}}, {concat});
    }

    // The producer can write into a view of the super buffer only when it
    // owns its whole allocation
    bool can_write_inplace(instruction_ref producer) const
    {
        auto alias = get_output_alias(producer);
        return is_allocation(alias) and alias->get_shape().lens() == producer->get_shape().lens();
    }

    struct plan
    {
        instruction_ref producer;
        std::vector<operation> view_ops = {};
        shape view_shape                = {};
    };

    // Decide if the producer of this concat input can write directly into a
    // view of the super buffer shaped like slice_shape
    std::optional<plan> plan_copy_elision(instruction_ref concat,
                                          instruction_ref input,
                                          const shape& slice_shape,
                                          std::size_t axis) const
    {
        auto [producer, chain] = get_view_chain(input);
        if(not can_write_inplace(producer))
            return std::nullopt;
        if(not slice_shape.packed() and not opt->supports_non_packed_output(producer, axis))
            return std::nullopt;
        // other users just read the strided view; the caller checks that
        // they support non-packed inputs
        if(chain.empty())
        {
            if(not producer_reports_shape(producer, slice_shape) or
               moves_output_layout(concat, producer, slice_shape))
                return std::nullopt;
            return plan{producer, {}, slice_shape};
        }
        if(producer->outputs().size() != 1)
            return std::nullopt;
        auto inverse = invert_view_chain(chain, producer->get_shape(), slice_shape);
        if(not inverse.has_value())
            return std::nullopt;
        if(not producer_reports_shape(producer, inverse->second) or
           moves_output_layout(concat, producer, inverse->second))
            return std::nullopt;
        return plan{producer, inverse->first, inverse->second};
    }

    // The producer must report the view's shape as its output so the reported
    // and actual layouts stay in sync; when the layouts already match nothing
    // needs to change, otherwise the target must be able to pin the shape
    bool producer_reports_shape(instruction_ref producer, const shape& view_shape) const
    {
        if(producer->get_shape() == view_shape)
            return true;
        return opt->supports_output_shape(producer, view_shape);
    }

    instruction_ref insert_copy(const operation& op,
                                instruction_ref concat,
                                instruction_ref input,
                                instruction_ref super,
                                std::size_t axis) const
    {
        auto slice = m->insert_instruction(std::next(super), op, super);
        if(auto p = plan_copy_elision(concat, input, slice->get_shape(), axis))
        {
            // Pin the layout before the allocation is replaced so the
            // producer never reports a shape out of sync with its buffer
            if(p->producer->get_shape() != p->view_shape)
                opt->set_output_shape(p->producer, p->view_shape);
            auto view = slice;
            for(const auto& vop : p->view_ops)
                view = m->insert_instruction(std::next(view), vop, view);
            m->replace_instruction(get_output_alias(p->producer), view);
            return input;
        }
        auto copy = m->insert_instruction(std::next(input), make_op(am.copy()), input, slice);
        // Other users read the copy, unless that would make a returned value non-standard
        if(not makes_output_non_standard({{input, copy->get_shape()}}, {concat, copy}))
            m->replace_instruction(input, copy);
        return copy;
    }

    // The shape a slice of the super buffer would have for this input
    static shape slice_shape_for(const shape& super_shape, const shape& input_shape)
    {
        return shape{super_shape.type(), input_shape.lens(), super_shape.strides()};
    }

    // The axis of a concat that writes into its own allocation
    std::optional<std::size_t> get_concat_axis(instruction_ref ins) const
    {
        auto concat_op = opt->get_concat(ins->get_operator());
        if(not concat_op.has_value() or not is_allocation(ins->inputs().back()))
            return std::nullopt;
        return tune_axis(
            ins->inputs().front()->get_shape().ndim(), concat_op->axis, concat_op->name());
    }

    static std::vector<instruction_ref> data_inputs(instruction_ref ins)
    {
        return {ins->inputs().begin(), std::prev(ins->inputs().end())};
    }

    // The position in the inputs of `ins` where the inputs of the smaller concat `sub` appear
    // as a consecutive run
    static std::optional<std::size_t> find_run(instruction_ref ins, instruction_ref sub)
    {
        auto inputs = data_inputs(ins);
        auto run    = data_inputs(sub);
        if(run.size() >= inputs.size())
            return std::nullopt;
        auto it = std::search(inputs.begin(), inputs.end(), run.begin(), run.end());
        if(it == inputs.end())
            return std::nullopt;
        std::size_t pos = std::distance(inputs.begin(), it);
        return pos;
    }

    // The other concats along the same axis that read the input
    std::vector<instruction_ref>
    concat_users(instruction_ref input, instruction_ref ins, std::size_t axis) const
    {
        std::vector<instruction_ref> result;
        std::copy_if(input->outputs().begin(),
                     input->outputs().end(),
                     std::back_inserter(result),
                     [&](instruction_ref out) {
                         return out != ins and m->has_instruction(out) and
                                get_concat_axis(out) == axis;
                     });
        return result;
    }

    bool can_replace(instruction_ref ins, std::size_t axis) const
    {
        auto inputs  = data_inputs(ins);
        auto ncopies = std::count_if(inputs.begin(), inputs.end(), [&](instruction_ref input) {
            auto slice_shape = slice_shape_for(ins->get_shape(), input->get_shape());
            return not plan_copy_elision(ins, input, slice_shape, axis).has_value();
        });
        if(ncopies > 1)
            return false;
        // The other consumers must support non-packed inputs
        return std::none_of(inputs.begin(), inputs.end(), [&](instruction_ref input) {
            if(input->outputs().size() < 2)
                return false;
            auto slice_shape = slice_shape_for(ins->get_shape(), input->get_shape());
            if(is_packed(slice_shape, axis))
                return false;
            return std::any_of(
                input->outputs().begin(), input->outputs().end(), [&](instruction_ref output) {
                    if(output == ins)
                        return false;
                    return not opt->supports_non_packed_input(output, axis);
                });
        });
    }

    // The largest replaceable concat that has every input of this concat as a consecutive run
    // of its own inputs, so the smaller concats become views of its buffer instead of each
    // claiming the shared inputs for a buffer of their own
    std::optional<instruction_ref> find_root(instruction_ref ins, std::size_t axis) const
    {
        auto users = concat_users(ins->inputs().front(), ins, axis);
        std::vector<instruction_ref> candidates;
        std::copy_if(users.begin(),
                     users.end(),
                     std::back_inserter(candidates),
                     [&](instruction_ref user) { return find_run(user, ins).has_value(); });
        candidates.push_back(ins);
        std::stable_sort(
            candidates.begin(), candidates.end(), by(std::greater<>{}, [](instruction_ref c) {
                return c->inputs().size();
            }));
        auto it = std::find_if(candidates.begin(), candidates.end(), [&](instruction_ref c) {
            return can_replace(c, axis);
        });
        if(it == candidates.end())
            return std::nullopt;
        return *it;
    }

    // A concat whose inputs are the `size` inputs of a larger concat starting at `pos`
    struct sub_concat
    {
        instruction_ref ins;
        std::size_t pos  = 0;
        std::size_t size = 0;
        operation slice  = {};
    };

    // The users of the concat can read it as a view of a larger buffer
    bool can_alias(instruction_ref sub, const shape& view_shape, std::size_t axis) const
    {
        if(not is_packed(view_shape, axis) and
           not std::all_of(sub->outputs().begin(), sub->outputs().end(), [&](instruction_ref out) {
               return opt->supports_non_packed_input(out, axis);
           }))
            return false;
        return not makes_output_non_standard({{sub, view_shape}}, {});
    }

    std::vector<sub_concat> find_sub_concats(instruction_ref ins, std::size_t axis) const
    {
        auto inputs = data_inputs(ins);
        std::vector<std::size_t> ends(inputs.size());
        transform_partial_sum(
            inputs.begin(), inputs.end(), ends.begin(), std::plus<>{}, [&](instruction_ref input) {
                return input->get_shape().lens()[axis];
            });
        std::vector<std::size_t> positions(inputs.size());
        std::iota(positions.begin(), positions.end(), 0);
        std::vector<sub_concat> result;
        std::transform(
            positions.begin(), positions.end(), join_back_inserter(result), [&](std::size_t pos) {
                auto users = concat_users(inputs[pos], ins, axis);
                auto start = ends[pos] - inputs[pos]->get_shape().lens()[axis];
                std::vector<sub_concat> subs;
                transform_if(
                    users.begin(),
                    users.end(),
                    std::back_inserter(subs),
                    [&](instruction_ref user) {
                        // An input can repeat, so only take the concat at its run
                        if(find_run(ins, user) != pos)
                            return false;
                        return can_alias(
                            user, slice_shape_for(ins->get_shape(), user->get_shape()), axis);
                    },
                    [&](instruction_ref user) {
                        return sub_concat{
                            user, pos, user->inputs().size() - 1, make_slice(user, axis, start)};
                    });
                return subs;
            });
        return result;
    }

    // Replace the concat, along with the smaller concats over the same inputs, which become
    // views of its buffer
    void replace_concats(instruction_ref ins, std::size_t axis) const
    {
        auto subs  = find_sub_concats(ins, axis);
        auto args  = replace_concat(ins, axis);
        auto super = args.front();
        for(const auto& sub : subs)
        {
            auto view = m->insert_instruction(std::next(super), sub.slice, super);
            // The inputs written into the buffer are kept as inputs to order the view after them
            std::vector<instruction_ref> inputs = {view};
            auto first                          = args.begin() + 1 + sub.pos;
            inputs.insert(inputs.end(), first, first + sub.size);
            m->replace_instruction(sub.ins, make_op("identity"), inputs);
        }
    }

    std::vector<instruction_ref> replace_concat(instruction_ref ins, std::size_t axis) const
    {
        auto last = ins->inputs().back();
        // Where are the allocations for the tensors to be concatenated?
        std::vector<instruction_ref> allocations;

        std::transform(ins->inputs().begin(),
                       std::prev(ins->inputs().end()),
                       std::back_inserter(allocations),
                       [](instruction_ref input) {
                           // resolve through view chains so the super buffer is
                           // placed before every producer's allocation
                           return get_output_alias(get_view_chain(input).first);
                       });

        // Need to sort the allocations, so that we know where to
        // insert the "super"-allocation
        auto sorted_allocations = allocations;
        std::sort(sorted_allocations.begin(),
                  sorted_allocations.end(),
                  [&](instruction_ref x, instruction_ref y) {
                      return std::distance(m->begin(), x) < std::distance(m->begin(), y);
                  });
        // Move "super" allocation to the front
        auto first                        = sorted_allocations.front();
        auto super                        = m->move_instruction(last, first);
        std::vector<instruction_ref> args = {super};
        std::size_t start                 = 0;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end() - 1,
                       std::back_inserter(args),
                       [&](instruction_ref input) {
                           auto x =
                               insert_copy(make_slice(input, axis, start), ins, input, super, axis);
                           start += x->get_shape().lens()[axis];
                           return x;
                       });
        m->replace_instruction(ins, migraphx::make_op("identity"), args);
        return args;
    }
};

} // namespace

void eliminate_concat::apply(module& m) const
{
    concat_optimizer co{&m, concat_opt.allocation(), &concat_opt};
    for(auto ins : iterator_for(m))
    {
        auto axis = co.get_concat_axis(ins);
        if(not axis.has_value())
            continue;
        auto root = co.find_root(ins, *axis);
        if(not root.has_value())
            continue;
        co.replace_concats(*root, *axis);
    }
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
