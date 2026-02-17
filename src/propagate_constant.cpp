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
#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/simple_par_for.hpp>
#include <migraphx/env.hpp>
#include <thread>
#include <unordered_set>

#if MIGRAPHX_USE_BLAZE
#include <migraphx/make_op.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_PROPAGATE_CONSTANT)

static bool skip_propagate(instruction_ref ins)
{
    if(contains({"contiguous", "dequantizelinear", "reshape"}, ins->name()))
        return skip_propagate(ins->inputs().front());
    if(contains({"unpack_int4", "unpack_fp4"}, ins->name()))
        return true;
    auto&& s = ins->get_shape();
    if(s.broadcasted() and s.element_space() < s.elements())
        return true;
    auto aliases = instruction::get_output_alias(ins, true);
    if(aliases.size() == 1 and aliases.front() != ins)
        return skip_propagate(aliases.front());
    if(ins->is_undefined())
        return true;
    return false;
}

static bool is_const_ins(instruction_ref ins, const std::unordered_set<std::string>& skip_ops)
{
    return ins->can_eval() and not skip_propagate(ins) and
           skip_ops.find(ins->name()) == skip_ops.end();
}

static argument as_packed(const argument& c)
{
    if(c.get_shape().packed())
        return c;
    auto s = c.get_shape().with_lens(c.get_shape().lens());
    argument result;
    c.visit([&](auto x) { result = literal{s, x.begin(), x.end()}.get_argument(); });
    return result;
}

#if MIGRAPHX_USE_BLAZE
// Rewrite constant 2D convolutions to im2col + dot in the graph.
// The existing propagate_constant loop will then fold these ops,
// with dot going through the Blaze-accelerated gemm.
static void rewrite_const_conv_to_im2col(module& m,
                                         const std::unordered_set<std::string>& skip_ops)
{
    std::vector<instruction_ref> const_convs;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "convolution" and is_const_ins(ins, skip_ops))
            const_convs.push_back(ins);
    }

    for(auto ins : const_convs)
    {
        auto conv_val = ins->get_operator().to_value();

        if(conv_val["group"].to<int>() != 1)
            continue;
        if(conv_val["padding_mode"].to<int>() != 0)
            continue;
        auto conv_stride = conv_val["stride"].to_vector<std::size_t>();
        if(conv_stride.size() != 2)
            continue;

        auto input  = ins->inputs()[0];
        auto weight = ins->inputs()[1];

        if(input->get_shape().ndim() != 4 or weight->get_shape().ndim() != 4)
            continue;
        if(input->get_shape().lens()[0] != 1)
            continue; // im2col only supports batch=1

        auto C_out     = weight->get_shape().lens()[0];
        auto C_in_kHkW = weight->get_shape().elements() / C_out;

        // im2col(input, weight) -> [outH*outW, C_in*kH*kW]
        auto col = m.insert_instruction(ins,
                                        make_op("im2col",
                                                {{"padding", conv_val["padding"]},
                                                 {"stride", conv_val["stride"]},
                                                 {"dilation", conv_val["dilation"]}}),
                                        input,
                                        weight);

        // reshape weight [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
        auto w_2d = m.insert_instruction(
            ins,
            make_op("reshape", {{"dims", {C_out, C_in_kHkW}}}),
            weight);

        // transpose weight -> [C_in*kH*kW, C_out]
        auto w_t = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {1, 0}}}), w_2d);

        auto w_contig = m.insert_instruction(ins, make_op("contiguous"), w_t);

        // dot: [outH*outW, C_in*kH*kW] x [C_in*kH*kW, C_out] = [outH*outW, C_out]
        auto result_2d = m.insert_instruction(ins, make_op("dot"), col, w_contig);

        // transpose -> [C_out, outH*outW]
        auto result_t = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {1, 0}}}), result_2d);

        auto result_contig = m.insert_instruction(ins, make_op("contiguous"), result_t);

        // reshape to conv output shape [1, C_out, outH, outW]
        auto out_lens = ins->get_shape().lens();
        m.replace_instruction(
            ins,
            make_op("reshape",
                    {{"dims",
                      std::vector<std::size_t>(out_lens.begin(), out_lens.end())}}),
            result_contig);
    }
}
#endif

void propagate_constant::apply(module& m) const
{
#if MIGRAPHX_USE_BLAZE
    // Rewrite constant convolutions to im2col + dot so the main loop
    // evaluates them through Blaze-accelerated gemm instead of naive conv.
    rewrite_const_conv_to_im2col(m, skip_ops);
#endif

    std::unordered_set<instruction_ref> const_instrs;
    auto last = std::prev(m.end());

    // Find instructions that can be evaluated to a literal
    for(auto i : iterator_for(m))
    {
        const bool is_const = is_const_ins(i, skip_ops);
        if(is_const and i != last)
            continue;

        if(i == last and is_const)
        {
            const_instrs.insert(i);
        }
        else
        {
            std::copy_if(i->inputs().begin(),
                         i->inputs().end(),
                         std::inserter(const_instrs, const_instrs.begin()),
                         [&](const instruction_ref ins) {
                             return is_const_ins(ins, skip_ops) and ins->name() != "@literal";
                         });
        }
    }

    // Compute literals in parallel
    std::vector<instruction_ref> const_instrs_vec{const_instrs.begin(), const_instrs.end()};
    std::vector<argument> literals(const_instrs_vec.size());
    std::size_t grainsize = 1;
#if !MIGRAPHX_HAS_EXECUTORS
    std::size_t n = std::max<std::size_t>(2048 / std::thread::hardware_concurrency(), 1);
    grainsize     = const_instrs_vec.size() / n;
#endif
    simple_par_for(const_instrs_vec.size(), grainsize, [&](const auto i) {
        literals[i] = as_packed(const_instrs_vec[i]->eval());
    });

    // Replace instructions in m
    for(size_t i = 0; i < const_instrs_vec.size(); i++)
    {
        if(not literals[i].empty())
        {
            if(enabled(MIGRAPHX_TRACE_PROPAGATE_CONSTANT{}))
            {
                std::cout << "Constant replace: " << std::endl;
                std::vector<instruction_ref> inss;
                fix([&](auto self, auto ins) {
                    if(contains(inss, ins))
                        return;
                    for(auto input : ins->inputs())
                        self(input);
                    inss.push_back(ins);
                })(const_instrs_vec[i]);
                m.debug_print(inss);
            }
            assert(literals[i].get_shape().lens() == const_instrs_vec[i]->get_shape().lens());
            assert(literals[i].get_shape().bytes() <= const_instrs_vec[i]->get_shape().bytes());
            auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
            m.replace_instruction(const_instrs_vec[i], l);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
