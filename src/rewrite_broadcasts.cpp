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
#include <migraphx/rewrite_broadcasts.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool is_valid_broadcast(const shape& s, std::vector<std::size_t> reduce_axes)
{
    const auto& blens    = s.lens();
    const auto& bstrides = s.strides();
    reduce_axes.erase(std::remove_if(reduce_axes.begin(),
                                     reduce_axes.end(),
                                     [&](std::size_t axis) { return blens.at(axis) == 1; }),
                      reduce_axes.end());

    std::vector<std::size_t> broadcast_axes;
    copy_if(range(bstrides.size()), std::back_inserter(broadcast_axes), [&](std::size_t i) {
        return bstrides.at(i) == 0 and blens.at(i) != 1;
    });

    return broadcast_axes == reduce_axes;
}

bool has_spanning_input(const std::vector<instruction_ref>& inputs,
                        instruction_ref broadcast,
                        const std::vector<std::size_t>& reduce_axes)
{
    const auto& blens = broadcast->get_shape().lens();
    return any_of(inputs, [&](instruction_ref input) {
        if(input == broadcast)
            return false;
        if(input->get_shape().dynamic())
            return false;
        const auto& lens = input->get_shape().lens();
        if(lens.size() != blens.size())
            return false;
        return all_of(reduce_axes,
                      [&](std::size_t axis) { return lens.at(axis) == blens.at(axis); });
    });
}

static std::vector<std::size_t> get_reduce_axes(instruction_ref ins)
{
    auto v = ins->get_operator().to_value();
    if(not v.contains("axes"))
        return {};
    return v.at("axes").to_vector<std::size_t>();
}

static instruction_ref
insert_ops(module& m, instruction_ref pos, const std::vector<operation>& ops, instruction_ref input)
{
    return std::accumulate(ops.begin(), ops.end(), input, [&](auto start, const auto& op) {
        return m.insert_instruction(pos, op, start);
    });
}

namespace {
struct pointwise_broadcast_op : match::supports_dynamic_shapes
{
    std::string op;

    auto matcher() const
    {
        auto pointwise = match::name("pointwise")(match::used_once()).bind("x");
        auto reshapes =
            match::name("reshape", "squeeze", "unsqueeze", "flatten", "transpose", "contiguous")(
                match::used_once());
        auto reshapes_pointwise = match::skip(reshapes)(pointwise);
        auto broadcast_pointwise =
            match::name("multibroadcast", "broadcast")(
                match::used_once(), match::nargs(1), match::arg(0)(reshapes_pointwise))
                .bind("broadcast");
        auto dyn_broadcast_pointwise =
            match::name("multibroadcast", "broadcast")(match::used_once(),
                                                       match::nargs(2),
                                                       match::arg(0)(reshapes_pointwise),
                                                       match::arg(1)(match::any().bind("ref_ins")))
                .bind("broadcast");
        return match::name(op)(match::any_of[match::inputs()](
            match::any_of(broadcast_pointwise, dyn_broadcast_pointwise)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto broadcast_ins    = r.instructions["broadcast"];
        auto x_ins            = r.instructions["x"];
        bool is_dyn_broadcast = contains(r.instructions, "ref_ins");

        auto broadcast = broadcast_ins->get_operator();

        // Reshapes between the pointwise and the broadcast apply to each input
        // as well since every pointwise input has the same lens as its output
        std::vector<operation> reshape_ops;
        auto next_ins = broadcast_ins->inputs().front();
        while(next_ins != x_ins)
        {
            reshape_ops.push_back(next_ins->get_operator());
            next_ins = next_ins->inputs().front();
        }
        std::reverse(reshape_ops.begin(), reshape_ops.end());

        // When the consumer reduces over exactly the axes the broadcast expands,
        // hoisting the pointwise past the broadcast would recompute it across the
        // reduction. Keep it at the pre-broadcast shape instead and emit an
        // ndim-matched multibroadcast which the reduce fusion can fuse directly.
        if(not is_dyn_broadcast and not broadcast_ins->get_shape().dynamic())
        {
            auto reduce_axes = get_reduce_axes(r.result);
            if(not reduce_axes.empty() and
               is_valid_broadcast(broadcast_ins->get_shape(), reduce_axes) and
               has_spanning_input(r.result->inputs(), broadcast_ins, reduce_axes))
            {
                const auto& bshape = broadcast_ins->get_shape();
                std::vector<std::size_t> mlens;
                std::transform(bshape.lens().begin(),
                               bshape.lens().end(),
                               bshape.strides().begin(),
                               std::back_inserter(mlens),
                               [](std::size_t len, std::size_t stride) {
                                   return stride == 0 ? std::size_t{1} : len;
                               });
                // Already in a form the reduce fusion can fuse directly
                if(reshape_ops.empty() and
                   broadcast_ins->inputs().front()->get_shape().lens() == mlens)
                    return;
                auto x_inputs = x_ins->inputs();
                std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), [&](auto input) {
                    input = insert_ops(m, broadcast_ins, reshape_ops, input);
                    if(input->get_shape().lens() == mlens)
                        return input;
                    return m.insert_instruction(
                        broadcast_ins, make_op("reshape", {{"dims", mlens}}), input);
                });
                auto pw = m.insert_instruction(
                    broadcast_ins, x_ins->get_operator(), x_inputs, x_ins->module_inputs());
                m.replace_instruction(
                    broadcast_ins, make_op("multibroadcast", {{"out_lens", bshape.lens()}}), pw);
                return;
            }
        }

        auto x_inputs = x_ins->inputs();
        std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), [&](auto input) {
            input = insert_ops(m, broadcast_ins, reshape_ops, input);
            if(is_dyn_broadcast)
            {
                return m.insert_instruction(
                    broadcast_ins, broadcast, {input, r.instructions["ref_ins"]});
            }
            return m.insert_instruction(broadcast_ins, broadcast, input);
        });

        m.replace_instruction(
            broadcast_ins, x_ins->get_operator(), x_inputs, x_ins->module_inputs());
    }
};
} // namespace

void rewrite_broadcasts(module_pass_manager& mpm, const std::string& op)
{
    pointwise_broadcast_op pbo;
    pbo.op = op;
    match::find_matches(mpm.get_module(), pbo);
    mpm.run_pass(eliminate_common_subexpression{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
