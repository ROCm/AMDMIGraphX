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

#include <migraphx/dfor.hpp>
#include <migraphx/tune_axis.hpp>
#include <iterator>
#include <numeric>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// shape_transform_descriptor only records the non-lazy reshape
shape_transform_descriptor make_view_descriptor(const std::vector<operation>& chain,
                                                const shape& input_shape)
{
    std::vector<operation> fwd;
    std::transform(chain.begin(), chain.end(), std::back_inserter(fwd), [](const operation& op) {
        if(op.name() == "reshape_lazy")
            return make_op("reshape", {{"dims", op.to_value().at("dims")}});
        return op;
    });
    return shape_transform_descriptor::create(input_shape.lens(), fwd);
}

} // namespace

optional<std::pair<std::vector<operation>, shape>> invert_view_chain(
    const std::vector<operation>& chain, const shape& input_shape, const shape& view_shape)
{
    auto inv = make_view_descriptor(chain, input_shape).invert();
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

// The axis of the chain's input that the chain maps as a whole onto the
// given axis of its output, so slicing either one selects the same elements
optional<std::size_t>
find_input_axis(const std::vector<operation>& chain, const shape& input_shape, std::size_t axis)
{
    auto desc = make_view_descriptor(chain, input_shape);
    if(desc.empty())
        return nullopt;
    auto axes = range(input_shape.ndim());
    auto it   = std::find_if(axes.begin(), axes.end(), [&](std::size_t i) {
        return desc.get_dst_axes_from_src(i) == std::vector<std::size_t>{axis};
    });
    if(it == axes.end())
        return nullopt;
    return *it;
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
    std::optional<plan>
    plan_copy_elision(instruction_ref input, const shape& slice_shape, std::size_t axis) const
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
            if(not producer_reports_shape(producer, slice_shape))
                return std::nullopt;
            return plan{producer, {}, slice_shape};
        }
        auto inverse = invert_view_chain(chain, producer->get_shape(), slice_shape);
        if(not inverse.has_value())
            return std::nullopt;
        if(not producer_reports_shape(producer, inverse->second))
            return std::nullopt;
        if(producer->outputs().size() != 1 and
           not users_support_view(producer, chain, inverse->second, axis))
            return std::nullopt;
        return plan{producer, inverse->first, inverse->second};
    }

    // The other users of a producer behind a view chain read the inverted
    // view of the slice, which is sliced along the producer axis that the
    // chain maps onto the concat axis
    bool users_support_view(instruction_ref producer,
                            const std::vector<operation>& chain,
                            const shape& view_shape,
                            std::size_t axis) const
    {
        auto producer_axis = find_input_axis(chain, producer->get_shape(), axis);
        if(not producer_axis.has_value())
            return false;
        if(is_packed(view_shape, *producer_axis))
            return true;
        return std::all_of(
            producer->outputs().begin(), producer->outputs().end(), [&](instruction_ref output) {
                return opt->supports_non_packed_input(output, *producer_axis);
            });
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
                                instruction_ref input,
                                instruction_ref super,
                                std::size_t axis) const
    {
        auto slice = m->insert_instruction(std::next(super), op, super);
        if(auto p = plan_copy_elision(input, slice->get_shape(), axis))
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
        m->replace_instruction(input, copy);
        return copy;
    }

    // The shape a slice of the super buffer would have for this input
    static shape slice_shape_for(const shape& super_shape, const shape& input_shape)
    {
        return shape{super_shape.type(), input_shape.lens(), super_shape.strides()};
    }

    void replace_concat(instruction_ref ins, std::size_t axis) const
    {
        // Last input should be an allocation
        auto last = ins->inputs().back();
        if(not is_allocation(last))
            return;
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
                           auto x = insert_copy(make_slice(input, axis, start), input, super, axis);
                           start += x->get_shape().lens()[axis];
                           return x;
                       });
        m->replace_instruction(ins, migraphx::make_op("identity"), args);
    }
};

} // namespace

void eliminate_concat::apply(module& m) const
{
    concat_optimizer co{&m, concat_opt.allocation(), &concat_opt};
    for(auto ins : iterator_for(m))
    {
        auto concat_op = concat_opt.get_concat(ins->get_operator());
        // Look for the concat operator
        if(not concat_op.has_value())
            continue;
        auto lens        = ins->inputs().front()->get_shape().lens();
        std::size_t axis = tune_axis(lens.size(), concat_op->axis, concat_op->name());
        auto ncopies     = std::count_if(
            ins->inputs().begin(), std::prev(ins->inputs().end()), [&](instruction_ref input) {
                auto slice_shape =
                    concat_optimizer::slice_shape_for(ins->get_shape(), input->get_shape());
                return not co.plan_copy_elision(input, slice_shape, axis).has_value();
            });
        if(ncopies > 1)
            continue;
        // If the other consumers cant support non-packed inputs
        if(std::any_of(
               ins->inputs().begin(), std::prev(ins->inputs().end()), [&](instruction_ref input) {
                   if(input->outputs().size() < 2)
                       return false;
                   auto slice_shape =
                       concat_optimizer::slice_shape_for(ins->get_shape(), input->get_shape());
                   if(is_packed(slice_shape, axis))
                       return false;
                   return std::any_of(input->outputs().begin(),
                                      input->outputs().end(),
                                      [&](instruction_ref output) {
                                          if(output == ins)
                                              return false;
                                          return not concat_opt.supports_non_packed_input(output,
                                                                                          axis);
                                      });
               }))
            continue;
        co.replace_concat(ins, axis);
    }
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
