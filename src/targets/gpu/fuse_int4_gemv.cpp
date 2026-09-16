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
#include <migraphx/gpu/fuse_int4_gemv.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/env.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/shape.hpp>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_INT4_GEMV);

struct int4_gemv_op
{
    std::string name() const { return "gpu::int4_gemv"; }

    std::size_t N       = 0;
    std::size_t K       = 0;
    int block_k         = 32;
    bool has_zp         = false;
    bool has_bias       = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.N, "N"), f(self.K, "K"), f(self.block_k, "block_k"), f(self.has_zp, "has_zp"), f(self.has_bias, "has_bias"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // inputs: A(f16), B_packed(u8), scales(f16), [zp(u8)], [bias(f16)]
        auto a_shape = inputs.front();
        auto batch   = a_shape.lens().front();
        return shape{shape::half_type, {batch, 1, N}};
    }
};
MIGRAPHX_REGISTER_OP(int4_gemv_op);

namespace {

// Trace an instruction back through view/reshape ops to find the root data tensor.
instruction_ref trace_through_views(instruction_ref ins)
{
    static const std::set<std::string> view_ops = {
        "broadcast", "multibroadcast", "reshape", "contiguous", "unsqueeze", "squeeze", "flatten"};
    auto cur = ins;
    while(view_ops.count(cur->name()) > 0 and not cur->inputs().empty())
        cur = cur->inputs().front();
    return cur;
}

// Check if `ins` has `unpack_int4` in its transitive input chain (bounded depth).
std::optional<instruction_ref> find_unpack_int4_ancestor(instruction_ref ins, int max_depth = 8)
{
    if(max_depth <= 0)
        return std::nullopt;
    if(ins->name() == "unpack_int4")
        return ins;
    for(auto& input : ins->inputs())
    {
        auto result = find_unpack_int4_ancestor(input, max_depth - 1);
        if(result.has_value())
            return result;
    }
    return std::nullopt;
}

// Find the dequantizelinear (or pointwise containing dequantize) op in the chain between
// unpack_int4 and dot. Returns the op and its scale/zp inputs.
struct dequant_info
{
    instruction_ref scales;
    instruction_ref zp;        // empty if symmetric
    instruction_ref unpack;    // the unpack_int4 instruction
    int block_k = 32;
    bool has_zp = false;
};

std::optional<dequant_info> find_dequant_in_chain(instruction_ref dot_b_input)
{
    // Walk back from dot's B input through reshaping ops to find dequantizelinear/pointwise
    static const std::set<std::string> reshape_ops = {
        "transpose", "contiguous", "reshape", "multibroadcast", "broadcast", "flatten", "squeeze",
        "unsqueeze", "slice"};

    auto cur = dot_b_input;
    while(reshape_ops.count(cur->name()) > 0 and not cur->inputs().empty())
    {
        std::cerr << "[int4_gemv] chain walk: " << cur->name() << " -> " << cur->inputs().front()->name() << std::endl;
        cur = cur->inputs().front();
    }
    std::cerr << "[int4_gemv] chain ended at: " << cur->name() << std::endl;

    // cur should now be dequantizelinear or a pointwise wrapping it
    if(cur->name() == "dequantizelinear")
    {
        // dequantizelinear(x, scale, [zp])
        auto inputs = cur->inputs();
        auto x      = inputs[0];
        // x should trace back to unpack_int4
        auto unpack = find_unpack_int4_ancestor(x);
        if(not unpack.has_value())
            return std::nullopt;

        dequant_info info;
        info.unpack = unpack.value();

        auto packed_shape = info.unpack->inputs().front()->get_shape();
        auto K            = packed_shape.lens().back() * 2;

        // Trace scales back through broadcast/reshape to compact [N, k_blocks]
        auto scales_root = trace_through_views(inputs[1]);
        auto sr_shape    = scales_root->get_shape();
        // Find the last non-1 dimension (skip trailing unsqueeze dims)
        auto sr_lens = sr_shape.lens();
        std::size_t sr_last = 1;
        for(auto it = sr_lens.rbegin(); it != sr_lens.rend(); ++it)
        {
            if(*it > 1) { sr_last = *it; break; }
        }
        if(sr_shape.ndim() >= 2 and sr_last > 1 and sr_last < K)
        {
            info.scales  = scales_root;
            info.block_k = static_cast<int>(K / sr_last);
        }
        else
        {
            info.scales  = inputs[1];
            info.block_k = 32;
        }

        if(inputs.size() >= 3)
        {
            info.has_zp = true;
            auto zp_root = trace_through_views(inputs[2]);
            info.zp      = zp_root;
        }
        return info;
    }

    // dequantizelinear was fused into a pointwise module by fuse_pointwise.
    if(cur->name() == "pointwise")
    {
        auto inputs = cur->inputs();
        if(inputs.size() < 2 or inputs.size() > 3)
            return std::nullopt;

        auto x = inputs[0];
        auto unpack = find_unpack_int4_ancestor(x);
        if(not unpack.has_value())
            return std::nullopt;

        dequant_info info;
        info.unpack = unpack.value();

        auto packed_shape = info.unpack->inputs().front()->get_shape();
        auto K            = packed_shape.lens().back() * 2;

        auto scales_root  = trace_through_views(inputs[1]);
        auto sr_shape     = scales_root->get_shape();
        auto sr_lens2 = sr_shape.lens();
        std::size_t sr_last = 1;
        for(auto it = sr_lens2.rbegin(); it != sr_lens2.rend(); ++it)
        {
            if(*it > 1) { sr_last = *it; break; }
        }
        if(sr_shape.ndim() >= 2 and sr_last > 1 and sr_last < K)
        {
            info.scales  = scales_root;
            info.block_k = static_cast<int>(K / sr_last);
        }
        else
        {
            info.scales  = inputs[1];
            info.block_k = 32;
        }

        if(inputs.size() >= 3)
        {
            info.has_zp = true;
            auto zp_root = trace_through_views(inputs[2]);
            info.zp      = zp_root;
        }
        std::cerr << "[int4_gemv] dequant MATCHED: K=" << K << " block_k=" << info.block_k << " has_zp=" << info.has_zp
                  << " scales.shape=" << info.scales->get_shape()
                  << std::endl;
        return info;
    }

    return std::nullopt;
}

// Check if a pointwise instruction is a simple binary add.
static bool is_pointwise_add(instruction_ref pw_ins)
{
    if(pw_ins->name() != "pointwise")
        return false;
    auto* pm = pw_ins->module_inputs().front();
    // A simple add pointwise module: @param x0, @param x1, add(x0, x1), @return
    for(auto ins : iterator_for(*pm))
    {
        if(ins->name() == "@param" or ins->name() == "@return" or
           ins->name() == "@literal" or ins->name() == "multibroadcast" or
           ins->name() == "broadcast")
            continue;
        if(ins->name() != "add")
            return false;
    }
    return true;
}

// Find the bias operand in a pointwise-add: the input that is NOT dot_ins.
// Returns nullopt if the add has more than 2 inputs or neither input is dot_ins.
static std::optional<instruction_ref> find_bias_in_add(instruction_ref pw_ins, instruction_ref dot_ins)
{
    auto inputs = pw_ins->inputs();
    if(inputs.size() != 2)
        return std::nullopt;
    if(inputs[0] == dot_ins)
        return inputs[1];
    if(inputs[1] == dot_ins)
        return inputs[0];
    return std::nullopt;
}

struct find_int4_gemv_op
{
    auto matcher() const
    {
        return match::name("dot");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto dot_ins = r.result;

        // Only intercept when the env var is set
        std::cerr << "[int4_gemv] matcher hit dot, env=" << enabled(MIGRAPHX_ENABLE_INT4_GEMV{}) << std::endl;
        if(not enabled(MIGRAPHX_ENABLE_INT4_GEMV{}))
            return;

        // Only static shapes for now
        if(dot_ins->get_shape().dynamic())
            return;

        // Check M == 1
        auto a_shape = dot_ins->inputs().front()->get_shape();
        if(a_shape.lens().size() < 2)
            return;
        auto m = a_shape.lens()[a_shape.lens().size() - 2];
        auto g = std::accumulate(
            a_shape.lens().begin(), a_shape.lens().end() - 2, std::size_t{1}, std::multiplies<>{});
        if(g * m != 1)
        {
            std::cerr << "[int4_gemv] skip: g*m=" << g*m << std::endl;
            return;
        }

        // Check that B input chain contains unpack_int4 and dequantizelinear
        auto b_input     = dot_ins->inputs().back();
        std::cerr << "[int4_gemv] M=1 dot found, B input=" << b_input->name() << std::endl;
        auto dequant     = find_dequant_in_chain(b_input);
        if(not dequant.has_value())
        {
            std::cerr << "[int4_gemv] skip: no dequant in chain" << std::endl;
            return;
        }
        std::cerr << "[int4_gemv] MATCHED! N=" << dot_ins->get_shape().lens().back() << std::endl;

        auto& info = dequant.value();

        // Get packed weights (input to unpack_int4)
        auto b_packed = info.unpack->inputs().front();
        auto K        = b_packed->get_shape().lens().back() * 2;
        auto N        = dot_ins->get_shape().lens().back();
        auto batch    = a_shape.lens().front();

        // Check for a trailing bias-add: dot has a single user that is pointwise(add)
        bool fuse_bias = false;
        instruction_ref add_ins;
        instruction_ref bias_input;
        auto dot_outputs = dot_ins->outputs();
        if(dot_outputs.size() == 1 and is_pointwise_add(dot_outputs.front()))
        {
            auto maybe_bias = find_bias_in_add(dot_outputs.front(), dot_ins);
            if(maybe_bias.has_value())
            {
                fuse_bias  = true;
                add_ins    = dot_outputs.front();
                bias_input = maybe_bias.value();
                std::cerr << "[int4_gemv] fusing trailing add (bias shape="
                          << bias_input->get_shape() << ")" << std::endl;
            }
        }

        // Build the replacement: gpu::int4_gemv(A, B_packed, scales, [zp], [bias])
        auto a_input = dot_ins->inputs().front();

        int4_gemv_op op;
        op.N        = N;
        op.K        = K;
        op.block_k  = info.block_k;
        op.has_zp   = info.has_zp;
        op.has_bias = fuse_bias;

        // The kernel takes zero-points already expanded to one u8 per element, so the ZP's
        // own unpack_int4 becomes a live input here and survives DCE as a standalone kernel
        // launch on every decode step -- 128 of them on a 32-layer asymmetric model.
        // propagate_constant deliberately skips unpack_int4 because folding the *weights*
        // would double their footprint; that argument does not apply to zero-points, which
        // are only N*k_blocks.  So fold just this one, when it is constant.
        if(info.has_zp and info.zp->name() == "unpack_int4" and info.zp->can_eval())
        {
            auto folded = info.zp->eval();
            if(not folded.empty() and folded.get_shape().standard())
            {
                std::cerr << "[int4_gemv] folded zp unpack_int4 -> literal "
                          << folded.get_shape() << std::endl;
                info.zp = mpm.get_module().add_literal(
                    literal{folded.get_shape(), folded.data()});
            }
        }

        std::vector<instruction_ref> new_inputs = {a_input, b_packed, info.scales};
        if(info.has_zp)
            new_inputs.push_back(info.zp);
        if(fuse_bias)
            new_inputs.push_back(bias_input);

        if(fuse_bias)
        {
            // Replace the add instruction (which consumes the dot) with the fused op
            mpm.get_module().replace_instruction(add_ins, op, new_inputs);
        }
        else
        {
            mpm.get_module().replace_instruction(dot_ins, op, new_inputs);
        }
    }
};


} // namespace

void fuse_int4_gemv::apply(module_pass_manager& mpm) const
{
    // INT4 M=1 GEMV: intercept before MLIR fuses the dot.
    // Controlled by MIGRAPHX_ENABLE_INT4_GEMV=1 (opt-in).
    match::find_matches(mpm, find_int4_gemv_op{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
