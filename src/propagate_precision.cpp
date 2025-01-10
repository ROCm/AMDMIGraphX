/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#include <migraphx/propagate_precision.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <unordered_set>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
// Class wrappper so we can compare precision using comparison operators
struct precision
{
    shape::type_t type;

    friend bool operator==(const precision& xp, const precision& yp) { return xp.type == yp.type; }
    friend bool operator<(const precision& xp, const precision& yp)
    {
        bool is_less = false;
        shape::visit(xp.type, [&](auto x) {
            shape::visit(yp.type, [&](auto y) {
                if(x.is_integral() != y.is_integral())
                    return;
                if(x.is_integral())
                {
                    if(x.is_unsigned() != y.is_unsigned() and x.size() == y.size())
                        is_less = y.is_unsigned();
                    else
                        is_less = x.size() < y.size();
                }
                else
                {
                    is_less = x.size() < y.size();
                }
            });
        });
        return is_less;
    }
    friend bool operator!=(const precision& xp, const precision& yp) { return not(xp == yp); }
    friend bool operator>(const precision& xp, const precision& yp) { return yp < xp; }
    // This is not totally ordered
    friend bool operator<=(const precision& xp, const precision& yp)
    {
        return (xp < yp) or (xp == yp);
    }
    friend bool operator>=(const precision& xp, const precision& yp)
    {
        return (xp > yp) or (xp == yp);
    }
};
#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace

static bool is_pointwise_or_reduce(instruction_ref ins)
{
    return contains(ins->name(), "reduce") or
           ins->get_operator().attributes().get("pointwise", false);
}
// Check if its not a scalar constant
static bool is_non_scalar_const(instruction_ref ins)
{
    return not(ins->get_shape().scalar() and ins->can_eval());
}
// Get the next input instruction otherwise return a nullopt
static std::optional<instruction_ref> get_next_input(instruction_ref ins)
{
    if(ins->inputs().size() == 1)
        return ins->inputs().front();
    if(ins->inputs().size() > 1)
    {
        std::unordered_set<instruction_ref> non_scalars;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::inserter(non_scalars, non_scalars.end()),
                     &is_non_scalar_const);
        if(non_scalars.size() == 1)
            return *non_scalars.begin();
    }
    return nullopt;
}

// Find all adjacent instructions that could be upgraded with higher precision
// by traversing the inputs from a convert

static std::unordered_set<instruction_ref> find_adjacent_inputs(instruction_ref start)
{
    std::unordered_set<instruction_ref> result;
    // Promote inputs
    fix([&](auto self, instruction_ref ins) {
        if(not is_pointwise_or_reduce(ins))
            return;
        if(contains(result, ins))
            return;
        auto next = get_next_input(ins);
        if(not next.has_value())
            return;
        result.insert(ins);
        self(*next);
    })(start->inputs().front());
    return result;
}

// Find all adjacent instructions that could be upgraded with higher precision
// by traversing the outputs from a convert
static std::unordered_set<instruction_ref> find_adjacent_outputs(instruction_ref start)
{
    std::unordered_set<instruction_ref> result;
    // Promote outputs
    fix([&](auto self, instruction_ref ins) {
        for(auto output : ins->outputs())
        {
            if(not is_pointwise_or_reduce(output))
                continue;
            if(contains(result, output))
                continue;
            auto next = get_next_input(output);
            if(not next.has_value())
                continue;
            if(*next != ins)
                continue;
            result.insert(output);
            self(output);
        }
    })(start);
    return result;
}

// Insert the instructions to upgrade into the map. If the map already has the
// instruction then choose the highest precision
template <class Map, class Instructions>
static void
insert_instructions_to_upgrade(Map& m, const Instructions& instructions, shape::type_t t)
{
    for(auto ins : instructions)
    {
        auto it = m.find(ins);
        if(it == m.end())
        {
            m[ins] = t;
        }
        else
        {
            it->second = std::max(precision{t}, precision{it->second}).type;
        }
    }
}

// Find adjacent instructions from a convert to upgrade to use a higher
// precision
static std::unordered_map<instruction_ref, shape::type_t> find_instruction_to_upgrade(module& m)
{
    std::unordered_map<instruction_ref, shape::type_t> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convert")
            continue;
        auto output = precision{ins->get_shape().type()};
        auto input  = precision{ins->inputs().front()->get_shape().type()};
        if(output.type == shape::type_t::bool_type)
            continue;
        if(input < output)
        {
            insert_instructions_to_upgrade(result, find_adjacent_inputs(ins), output.type);
        }
        else if(input > output)
        {
            insert_instructions_to_upgrade(result, find_adjacent_outputs(ins), input.type);
        }
    }
    return result;
}

void propagate_precision::apply(module_pass_manager& mpm) const
{
    auto upgrade = find_instruction_to_upgrade(mpm.get_module());
    for(const auto& p : upgrade)
    {
        auto ins      = p.first;
        auto t        = p.second;
        auto convert1 = mpm.get_module().insert_instruction(
            std::next(ins), make_op("convert", {{"target_type", ins->get_shape().type()}}), ins);
        mpm.get_module().replace_instruction(ins, convert1);
        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(inputs),
                       [&](auto input) {
                           return mpm.get_module().insert_instruction(
                               ins, make_op("convert", {{"target_type", t}}), input);
                       });
        mpm.get_module().replace_instruction(ins, ins->get_operator(), inputs);
    }
    mpm.run_pass(eliminate_convert{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
