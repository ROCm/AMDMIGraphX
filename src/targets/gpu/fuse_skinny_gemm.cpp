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
#include <migraphx/gpu/fuse_skinny_gemm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/env.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SKINNY_GEMM);

namespace {

constexpr std::size_t cols_per_thread = 8;
constexpr std::size_t block_size      = 256;
constexpr std::size_t max_m           = 8;
// Wide outputs already fill the GPU without split-K and run at bandwidth on
// the MFMA path, so only take over the narrow weight-streaming shapes.
constexpr std::size_t max_n = 8192;

struct skinny_gemm_splitk
{
    std::size_t splits = 1;
    // when set the skinny input is the packed {m, 2k} gate/up projection and
    // up * silu(gate) is applied while staging it
    bool swiglu = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.splits, "splits"), f(self.swiglu, "swiglu"));
    }

    std::string name() const { return "gpu::skinny_gemm_splitk"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& a = inputs.at(0);
        const auto& b = inputs.at(1);
        auto m        = a.elements() / a.lens().back();
        return shape{shape::float_type, {splits, m, b.lens().at(1)}};
    }
};
MIGRAPHX_REGISTER_OP(skinny_gemm_splitk);

struct skinny_gemm_reduce
{
    shape output_shape;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "gpu::skinny_gemm_reduce"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1, 2);
        return output_shape;
    }
};
MIGRAPHX_REGISTER_OP(skinny_gemm_reduce);

// Pick the split count so the grid covers the CUs about twice over; deeper
// splits shrink the main kernel but grow the partials reduction.
std::size_t compute_splits(std::size_t k, std::size_t n)
{
    auto ntiles = (n + block_size * cols_per_thread - 1) / (block_size * cols_per_thread);
    auto splits = std::max<std::size_t>(608 / ntiles, 8);
    splits      = std::min(splits, k / 16);
    return std::min<std::size_t>(splits, 320);
}

// Match the swiglu pointwise produced for the gate/up projection:
// mul(mul(x0, sigmoid(x0)), x1) with x0 and x1 the two halves of one tensor.
// Returns the packed gate/up tensor when it matches.
std::optional<instruction_ref> match_swiglu_input(instruction_ref a, std::size_t k)
{
    if(a->name() != "pointwise" or a->inputs().size() != 2 or a->outputs().size() != 1)
        return std::nullopt;
    auto* pm = a->module_inputs().front();
    auto ret = std::prev(pm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto x0 = pm->get_parameter("x0");
    auto x1 = pm->get_parameter("x1");
    if(x0 == pm->end() or x1 == pm->end())
        return std::nullopt;
    auto outer = ret->inputs().front();
    if(outer->name() != "mul")
        return std::nullopt;
    // out = mul(mul(gate, sigmoid(gate)), up) with gate/up bound to either
    // parameter
    instruction_ref inner = pm->end();
    instruction_ref up    = pm->end();
    for(auto input : outer->inputs())
    {
        if(input == x0 or input == x1)
            up = input;
        else
            inner = input;
    }
    if(up == pm->end() or inner == pm->end() or inner->name() != "mul")
        return std::nullopt;
    auto gate     = (up == x0) ? x1 : x0;
    bool has_gate = false;
    bool has_sig  = false;
    for(auto input : inner->inputs())
    {
        if(input == gate)
            has_gate = true;
        else if(input->name() == "sigmoid" and input->inputs().front() == gate)
            has_sig = true;
    }
    if(not has_gate or not has_sig)
        return std::nullopt;

    // the two inputs must be the adjacent halves of the same tensor, gate
    // first (parameters bind to the pointwise inputs in sorted name order)
    auto gate_sl = (gate == x0) ? a->inputs().at(0) : a->inputs().at(1);
    auto up_sl   = (gate == x0) ? a->inputs().at(1) : a->inputs().at(0);
    auto is_half = [&](instruction_ref sl, std::size_t start) {
        if(sl->name() != "slice")
            return false;
        auto v    = sl->get_operator().to_value();
        auto axes = v.at("axes").to_vector<int64_t>();
        if(axes.size() != 1)
            return false;
        auto rank = sl->inputs().front()->get_shape().lens().size();
        if(axes.front() != static_cast<int64_t>(rank - 1))
            return false;
        return v.at("starts").to_vector<int64_t>() ==
                   std::vector<int64_t>{static_cast<int64_t>(start)} and
               v.at("ends").to_vector<int64_t>() ==
                   std::vector<int64_t>{static_cast<int64_t>(start + k)};
    };
    if(not is_half(gate_sl, 0) or not is_half(up_sl, k))
        return std::nullopt;
    auto gu = gate_sl->inputs().front();
    if(up_sl->inputs().front() != gu)
        return std::nullopt;
    if(not gu->get_shape().standard() or gu->get_shape().lens().back() != 2 * k)
        return std::nullopt;
    return gu;
}

struct find_skinny_gemm
{
    auto matcher() const
    {
        return match::name("dot")(match::arg(1)(
            match::name("multibroadcast")(match::arg(0)(match::name("@literal").bind("w")))));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto dot = r.result;
        auto w   = r.instructions["w"];
        auto a   = dot->inputs().front();

        const auto& w_shape = w->get_shape();
        auto out_shape      = dot->get_shape();
        if(not w_shape.standard())
            return;
        if(w_shape.lens().size() != 2)
            return;
        auto k = w_shape.lens().front();
        auto n = w_shape.lens().back();

        bool swiglu = false;
        if(auto gu = match_swiglu_input(a, k))
        {
            swiglu = true;
            a      = *gu;
        }
        const auto& a_shape = a->get_shape();
        auto m_dim          = a_shape.elements() / a_shape.lens().back();
        if(not a_shape.standard())
            return;
        if(a_shape.lens().back() != (swiglu ? 2 * k : k))
            return;
        if(m_dim > max_m or n > max_n or n % cols_per_thread != 0)
            return;
        // only worth it in the weight-streaming regime
        if(k * n < (1 << 23))
            return;
        if(not contains({shape::bf16_type, shape::half_type, shape::float_type}, a_shape.type()))
            return;
        if(a_shape.type() != w_shape.type())
            return;

        // fold a single elementwise add user into the reduce step
        instruction_ref residual = m.end();
        auto ins                 = dot;
        if(dot->outputs().size() == 1)
        {
            auto user = dot->outputs().front();
            if(user->name() == "pointwise" and user->inputs().size() == 2 and
               user->get_shape() == out_shape)
            {
                auto* pm  = user->module_inputs().front();
                auto ret  = std::prev(pm->end());
                auto last = ret->inputs().front();
                if(ret->name() == "@return" and ret->inputs().size() == 1 and
                   last->name() == "add" and
                   std::all_of(last->inputs().begin(), last->inputs().end(), [](instruction_ref i) {
                       return i->name() == "@param";
                   }))
                {
                    auto other = user->inputs().front() == dot ? user->inputs().back()
                                                               : user->inputs().front();
                    if(other != dot and other->get_shape() == out_shape and
                       other->get_shape().standard())
                    {
                        residual = other;
                        ins      = user;
                    }
                }
            }
        }

        // Without a folded residual the separate reduce kernel is pure
        // overhead and the in-model MFMA path wins; measured on qwen3-8b
        // decode at both short and long context.
        if(residual == m.end())
            return;

        auto splits  = compute_splits(k, n);
        auto partial = m.insert_instruction(ins, skinny_gemm_splitk{splits, swiglu}, {a, w});
        std::vector<instruction_ref> reduce_inputs = {partial};
        if(residual != m.end())
            reduce_inputs.push_back(residual);
        m.replace_instruction(ins, skinny_gemm_reduce{out_shape}, reduce_inputs);
    }
};

} // namespace

void fuse_skinny_gemm::apply(module_pass_manager& mpm) const
{
    if(enabled(MIGRAPHX_DISABLE_SKINNY_GEMM{}))
        return;
    match::find_matches(mpm.get_module(), find_skinny_gemm{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
