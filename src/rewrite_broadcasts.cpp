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
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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

        auto x_inputs = x_ins->inputs();
        std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), [&](auto input) {
            input =
                std::accumulate(reshape_ops.begin(),
                                reshape_ops.end(),
                                input,
                                [&](auto start, const auto& reshape_op) {
                                    return m.insert_instruction(broadcast_ins, reshape_op, start);
                                });
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
