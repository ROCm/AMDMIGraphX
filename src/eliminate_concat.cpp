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
#include <migraphx/reshape_dims.hpp>

#include <migraphx/dfor.hpp>
#include <migraphx/tune_axis.hpp>
#include <iterator>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

bool reshape_lazy_accepts_input_shape(instruction_ref ins, const shape& input_shape)
{
    if(ins->name() != "reshape_lazy")
        return true;
    return reshape_dims(input_shape, ins->get_shape().lens(), {.lazy = true}).has_value();
}

struct concat_optimizer
{
    module* m = nullptr;
    allocation_model am;

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

    bool need_copy(instruction_ref ins, const shape& replacement_shape) const
    {
        auto alias = get_output_alias(ins);
        return not is_allocation(alias) or
               not replacement_supports_reshape_lazy(alias, replacement_shape);
    }

    bool replacement_supports_reshape_lazy(instruction_ref replaced,
                                           const shape& replacement_shape) const
    {
        std::unordered_map<instruction_ref, shape> shapes = {{replaced, replacement_shape}};
        auto downstream_range                             = range(std::next(replaced), m->end());
        auto downstream_refs                              = iterator_for(downstream_range);
        return std::all_of(
            downstream_refs.begin(), downstream_refs.end(), [&](instruction_ref ins) {
                bool changed      = false;
                auto input_shapes = to_shapes(ins->inputs());
                std::transform(ins->inputs().begin(),
                               ins->inputs().end(),
                               input_shapes.begin(),
                               [&](instruction_ref arg) {
                                   auto it = shapes.find(arg);
                                   if(it == shapes.end())
                                       return arg->get_shape();
                                   changed = true;
                                   return it->second;
                               });
                if(not changed)
                    return true;
                if(not reshape_lazy_accepts_input_shape(ins, input_shapes.front()))
                    return false;
                shape output_shape;
                try
                {
                    output_shape =
                        ins->get_operator().compute_shape(input_shapes, ins->module_inputs());
                }
                catch(const std::exception&)
                {
                    return false;
                }
                if(output_shape != ins->get_shape())
                    shapes.emplace(ins, std::move(output_shape));
                return true;
            });
    }

    instruction_ref
    insert_copy(const operation& op, instruction_ref input, instruction_ref super) const
    {
        auto slice = m->insert_instruction(std::next(super), op, super);
        // If its packed then replace the allocation with the slice instead
        if(not need_copy(input, slice->get_shape()) and slice->get_shape().packed() and
           input->outputs().size() == 1)
        {
            m->replace_instruction(get_output_alias(input), slice);
            return input;
        }
        auto copy = m->insert_instruction(std::next(input), make_op(am.copy()), input, slice);
        m->replace_instruction(input, copy);
        return copy;
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
                       &get_output_alias);

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
                           auto x = insert_copy(make_slice(input, axis, start), input, super);
                           start += x->get_shape().lens()[axis];
                           return x;
                       });
        m->replace_instruction(ins, migraphx::make_op("identity"), args);
    }
};

bool is_packed(instruction_ref ins, std::size_t axis)
{
    auto alens  = ins->get_shape().lens();
    alens[axis] = 1;
    return shape{ins->get_shape().type(), alens, ins->get_shape().strides()}.packed();
}

} // namespace

void eliminate_concat::apply(module& m) const
{
    concat_optimizer co{&m, concat_opt.allocation()};
    for(auto ins : iterator_for(m))
    {
        auto concat_op = concat_opt.get_concat(ins->get_operator());
        // Look for the concat operator
        if(not concat_op.has_value())
            continue;
        auto lens        = ins->inputs().front()->get_shape().lens();
        std::size_t axis = tune_axis(lens.size(), concat_op->axis, concat_op->name());
        if(std::any_of(
               ins->inputs().begin(), std::prev(ins->inputs().end()), [&](instruction_ref input) {
                   auto slice_shape = shape{input->get_shape().type(),
                                            input->get_shape().lens(),
                                            ins->get_shape().strides()};
                   auto needs_copy = co.need_copy(input, slice_shape) or not slice_shape.packed() or
                                     input->outputs().size() != 1;
                   return needs_copy and
                          not co.replacement_supports_reshape_lazy(input, slice_shape);
               }))
            continue;
        auto ncopies = std::count_if(
            ins->inputs().begin(), std::prev(ins->inputs().end()), [&](instruction_ref input) {
                auto slice_shape = shape{input->get_shape().type(),
                                         input->get_shape().lens(),
                                         ins->get_shape().strides()};
                if(co.need_copy(input, slice_shape))
                    return true;
                if(is_packed(input, axis))
                    return false;
                return not concat_opt.supports_non_packed_output(input, axis);
            });
        if(ncopies > 1)
            continue;
        // If the other consumers cant support non-packed inputs
        if(std::any_of(
               ins->inputs().begin(), std::prev(ins->inputs().end()), [&](instruction_ref input) {
                   if(input->outputs().size() < 2)
                       return false;
                   if(is_packed(input, axis))
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
