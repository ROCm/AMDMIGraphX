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
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SKINNY_GEMM);
// Fusing the rmsnorm into the partials reduce caps the kernel at one
// workgroup per row; the partials were written by workgroups on every XCD, so
// the cross-die reads run ~3x slower than the wide reduce + norm pair it
// replaces. Kept behind a flag for skinnier-partial cases.
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_SKINNY_RMSNORM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SKINNY_ROPE);

namespace {

constexpr std::size_t tile_n = 256;
constexpr std::size_t max_m  = 8;
// Wide outputs already fill the GPU without split-K and run at bandwidth on
// the rocMLIR path, so only take over the narrow weight-streaming shapes.
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

struct skinny_gemm_reduce_rmsnorm
{
    shape output_shape;
    float eps      = 1e-6f;
    float ss_scale = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"),
                    f(self.eps, "eps"),
                    f(self.ss_scale, "ss_scale"));
    }

    std::string name() const { return "gpu::skinny_gemm_reduce_rmsnorm"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // partials, residual -> (row, rmsnorm(row))
        check_shapes{inputs, *this}.has(2);
        return shape{{output_shape, output_shape}};
    }
};
MIGRAPHX_REGISTER_OP(skinny_gemm_reduce_rmsnorm);

// Pick the split count so the grid covers the CUs about four waves deep;
// deeper splits shrink the main kernel but grow the partials reduction.
std::size_t compute_splits(std::size_t k, std::size_t n)
{
    auto ntiles = (n + tile_n - 1) / tile_n;
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

std::optional<float> get_scalar_literal(instruction_ref ins)
{
    if(ins->name() != "@literal")
        return std::nullopt;
    if(ins->get_shape().elements() != 1)
        return std::nullopt;
    float result = 0;
    ins->get_literal().visit([&](auto data) { result = static_cast<float>(data[0]); });
    return result;
}

// x -> convert(float) -> x*x -> *scale ; returns scale
std::optional<float> check_square_scale_module(const_module_ref pm)
{
    auto ret = std::prev(pm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto outer = ret->inputs().front();
    if(outer->name() != "mul")
        return std::nullopt;
    instruction_ref inner = pm->end();
    std::optional<float> scale;
    for(auto input : outer->inputs())
    {
        if(auto lit = get_scalar_literal(input))
            scale = lit;
        else
            inner = input;
    }
    if(not scale.has_value() or inner == pm->end())
        return std::nullopt;
    if(inner->name() != "mul" or inner->inputs().at(0) != inner->inputs().at(1))
        return std::nullopt;
    auto cvt = inner->inputs().front();
    if(cvt->name() != "convert" or cvt->get_shape().type() != shape::float_type)
        return std::nullopt;
    if(cvt->inputs().front()->name() != "@param")
        return std::nullopt;
    return scale;
}

// x -> +eps -> rsqrt -> convert ; returns eps
std::optional<float> check_eps_rsqrt_module(const_module_ref pm)
{
    auto ret = std::prev(pm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto cvt = ret->inputs().front();
    if(cvt->name() != "convert")
        return std::nullopt;
    auto rsqrt = cvt->inputs().front();
    if(rsqrt->name() != "rsqrt")
        return std::nullopt;
    auto add = rsqrt->inputs().front();
    if(add->name() != "add")
        return std::nullopt;
    std::optional<float> eps;
    bool has_param = false;
    for(auto input : add->inputs())
    {
        if(auto lit = get_scalar_literal(input))
            eps = lit;
        else if(input->name() == "@param")
            has_param = true;
    }
    if(not has_param)
        return std::nullopt;
    return eps;
}

struct rmsnorm_info
{
    float eps      = 0;
    float ss_scale = 0;
};

// Match a gamma-less rmsnorm fused_reduce module:
//   pw_square_scale(x0) -> reduce_sum[last axis] -> pw_eps_rsqrt ->
//   multibroadcast -> mul(x0, rms)
std::optional<rmsnorm_info> match_rmsnorm(instruction_ref fr)
{
    if(fr->name() != "fused_reduce" or fr->inputs().size() != 1)
        return std::nullopt;
    auto v    = fr->get_operator().to_value();
    auto axes = v.at("axes").to_vector<int64_t>();
    auto rank = static_cast<int64_t>(fr->get_shape().lens().size());
    if(axes != std::vector<int64_t>{rank - 1})
        return std::nullopt;
    auto* rm = fr->module_inputs().front();

    auto ret = std::prev(rm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto apply = ret->inputs().front();
    if(apply->name() != "pointwise" or apply->inputs().size() != 2)
        return std::nullopt;
    // apply = mul(data, rms) in either operand order
    auto* am   = apply->module_inputs().front();
    auto aret  = std::prev(am->end());
    auto amul  = aret->inputs().front();
    if(aret->name() != "@return" or amul->name() != "mul")
        return std::nullopt;
    instruction_ref data = rm->end();
    instruction_ref mb   = rm->end();
    for(auto input : apply->inputs())
    {
        if(input->name() == "@param")
            data = input;
        else if(input->name() == "multibroadcast")
            mb = input;
    }
    if(data == rm->end() or mb == rm->end())
        return std::nullopt;
    auto rsqrt_pw = mb->inputs().front();
    if(rsqrt_pw->name() != "pointwise" or rsqrt_pw->inputs().size() != 1)
        return std::nullopt;
    auto eps = check_eps_rsqrt_module(rsqrt_pw->module_inputs().front());
    if(not eps.has_value())
        return std::nullopt;
    auto rsum = rsqrt_pw->inputs().front();
    if(rsum->name() != "reduce_sum")
        return std::nullopt;
    auto square_pw = rsum->inputs().front();
    if(square_pw->name() != "pointwise" or square_pw->inputs().size() != 1 or
       square_pw->inputs().front() != data)
        return std::nullopt;
    auto scale = check_square_scale_module(square_pw->module_inputs().front());
    if(not scale.has_value())
        return std::nullopt;
    return rmsnorm_info{*eps, *scale};
}

// Fuse a following gamma-less rmsnorm into the partials reduction: the fused
// kernel emits both the raw row (for the residual chain) and its norm
struct find_skinny_reduce_rmsnorm
{
    auto matcher() const
    {
        return match::name("fused_reduce")(
            match::arg(0)(match::name("gpu::skinny_gemm_reduce").bind("reduce")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto fr     = r.result;
        auto reduce = r.instructions["reduce"];
        // only the residual-folding form materializes the value the norm sees
        if(reduce->inputs().size() != 2)
            return;
        auto info = match_rmsnorm(fr);
        if(not info.has_value())
            return;
        const auto& out_shape = reduce->get_shape();
        if(not out_shape.standard() or out_shape.type() != shape::bf16_type)
            return;

        auto fused = m.insert_instruction(
            reduce,
            skinny_gemm_reduce_rmsnorm{out_shape, info->eps, info->ss_scale},
            reduce->inputs());
        auto h = m.insert_instruction(
            reduce, make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto norm = m.insert_instruction(
            reduce, make_op("get_tuple_elem", {{"index", 1}}), fused);
        m.replace_instruction(fr, norm);
        m.replace_instruction(reduce, h);
    }
};

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
        // MFMA column tiles are 16 wide
        if(m_dim > max_m or n > max_n or n % 16 != 0)
            return;
        // only worth it in the weight-streaming regime
        if(k * n < (1 << 23))
            return;
        // the split kernel computes with bf16 MFMA instructions
        if(a_shape.type() != shape::bf16_type or w_shape.type() != shape::bf16_type)
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

        auto splits  = compute_splits(k, n);
        auto partial = m.insert_instruction(ins, skinny_gemm_splitk{splits, swiglu}, {a, w});
        std::vector<instruction_ref> reduce_inputs = {partial};
        if(residual != m.end())
            reduce_inputs.push_back(residual);
        m.replace_instruction(ins, skinny_gemm_reduce{out_shape}, reduce_inputs);
    }
};

bool is_v_slice(instruction_ref ins, int64_t start, int64_t end)
{
    if(ins->name() != "slice")
        return false;
    auto v = ins->get_operator().to_value();
    return v.at("axes").to_vector<int64_t>() == std::vector<int64_t>{2} and
           v.at("starts").to_vector<int64_t>() == std::vector<int64_t>{start} and
           v.at("ends").to_vector<int64_t>() == std::vector<int64_t>{end};
}

// Feed the qkv split-K partials straight into the rope kernel: the rope
// workgroups sum the splits themselves (they are as parallel as the reduce
// was) and the v heads pass through as a second output, so the separate
// partials-reduce kernel disappears.
struct find_skinny_rope
{
    auto matcher() const
    {
        return match::name("gpu::rope_qk_norm")(
            match::arg(0)(match::name("gpu::skinny_gemm_reduce").bind("reduce")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto rope   = r.result;
        auto reduce = r.instructions["reduce"];
        // only the plain form; a folded residual has other consumers by design
        if(reduce->inputs().size() != 1)
            return;
        auto splitk = reduce->inputs().front();
        if(splitk->name() != "gpu::skinny_gemm_splitk")
            return;
        auto rv = rope->get_operator().to_value();
        if(rv.at("splits").to<std::size_t>() != 0)
            return;
        const auto nq = rv.at("num_heads").to<int64_t>();

        const auto& out_lens = rope->get_shape().lens(); // {b, nq+nk, 1, d}
        const int64_t b      = out_lens.at(0);
        const int64_t h      = out_lens.at(1);
        const int64_t d      = out_lens.at(3);
        const int64_t w      = reduce->get_shape().lens().back();
        const int64_t total  = w / d;
        const int64_t nk     = total - h;
        if(h != nq + nk or total != nq + 2 * nk or nk <= 0)
            return;
        if(d % 2 != 0)
            return;

        // every other consumer of the projection must be the v-head view
        // chain fuse_rope_qk_norm produced: slice -> reshape -> transpose
        std::vector<instruction_ref> v_users;
        for(auto out : reduce->outputs())
        {
            if(out == rope)
                continue;
            if(not is_v_slice(out, (nq + nk) * d, total * d))
                return;
            if(out->outputs().size() != 1)
                return;
            auto rsh = out->outputs().front();
            if(rsh->name() != "reshape" and rsh->name() != "reshape_lazy")
                return;
            if(rsh->outputs().size() != 1)
                return;
            auto tp = rsh->outputs().front();
            if(tp->name() != "transpose" or
               tp->get_operator().to_value().at("permutation").to_vector<int64_t>() !=
                   std::vector<int64_t>{0, 2, 1, 3})
                return;
            if(tp->get_shape().lens() !=
               std::vector<std::size_t>{static_cast<std::size_t>(b),
                                        static_cast<std::size_t>(nk),
                                        1,
                                        static_cast<std::size_t>(d)})
                return;
            v_users.push_back(tp);
        }

        const auto& p_shape = splitk->get_shape(); // {splits, b, total*d}
        if(p_shape.type() != shape::float_type)
            return;

        auto args = rope->inputs();
        args[0]   = splitk;

        // the fused op must sit before the first of its consumers (the rope
        // and the v chains can be in either order) and after all its inputs
        std::unordered_set<instruction_ref> uses(v_users.begin(), v_users.end());
        uses.insert(rope);
        std::unordered_set<instruction_ref> pending(args.begin(), args.end());
        auto first_use = m.end();
        for(auto it = m.begin(); it != m.end(); ++it)
        {
            pending.erase(it);
            if(contains(uses, it))
            {
                first_use = it;
                break;
            }
        }
        if(first_use == m.end() or not pending.empty())
            return;

        rv["splits"] = p_shape.lens().front();
        auto fused   = m.insert_instruction(first_use, make_op("gpu::rope_qk_norm", rv), args);
        for(auto tp : v_users)
        {
            auto v = m.insert_instruction(tp, make_op("get_tuple_elem", {{"index", 1}}), fused);
            m.replace_instruction(tp, v);
        }
        auto qk = m.insert_instruction(rope, make_op("get_tuple_elem", {{"index", 0}}), fused);
        m.replace_instruction(rope, qk);
    }
};

} // namespace

void fuse_skinny_gemm::apply(module_pass_manager& mpm) const
{
    if(enabled(MIGRAPHX_DISABLE_SKINNY_GEMM{}))
        return;
    match::find_matches(mpm.get_module(), find_skinny_gemm{});
    if(not enabled(MIGRAPHX_DISABLE_SKINNY_ROPE{}))
        match::find_matches(mpm.get_module(), find_skinny_rope{});
    if(enabled(MIGRAPHX_ENABLE_SKINNY_RMSNORM{}))
        match::find_matches(mpm.get_module(), find_skinny_reduce_rmsnorm{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
