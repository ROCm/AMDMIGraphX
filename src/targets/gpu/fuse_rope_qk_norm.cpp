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
#include <migraphx/gpu/fuse_rope_qk_norm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <optional>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

struct rope_qk_norm
{
    std::size_t num_heads = 0;
    float eps             = 1e-6f;
    float ss_scale        = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.num_heads, "num_heads"), f(self.eps, "eps"), f(self.ss_scale, "ss_scale"));
    }

    std::string name() const { return "gpu::rope_qk_norm"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // qkv, q_norm_weight, k_norm_weight, cos, signed_sin
        check_shapes{inputs, *this}.has(5);
        const auto& qkv = inputs.front();
        auto d          = inputs.at(1).lens().front();
        auto b          = qkv.lens().at(0);
        auto w          = qkv.lens().at(2);
        auto total      = w / d;
        auto nk         = (total - num_heads) / 2;
        auto h          = num_heads + nk;
        // Same layout the fused pointwise produced: a transposed view of
        // {b, 1, h, d}, so downstream shapes are unchanged.
        return shape{qkv.type(), {b, h, 1, d}, {h * d, d, h * d, 1}};
    }
};
MIGRAPHX_REGISTER_OP(rope_qk_norm);

std::unordered_map<instruction_ref, std::size_t> param_indices(const_module_ref pm)
{
    std::unordered_map<instruction_ref, std::size_t> result;
    auto names = pm->get_parameter_names();
    std::sort(names.begin(), names.end());
    for(auto i : range(names.size()))
        result[pm->get_parameter(names[i])] = i;
    return result;
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

bool is_slice(instruction_ref ins,
              const std::vector<int64_t>& axes,
              const std::vector<int64_t>& starts,
              const std::vector<int64_t>& ends)
{
    if(ins->name() != "slice")
        return false;
    auto v = ins->get_operator().to_value();
    return v.at("axes").to_vector<int64_t>() == axes and
           v.at("starts").to_vector<int64_t>() == starts and
           v.at("ends").to_vector<int64_t>() == ends;
}

// out = add(mul(x0, x1), mul(x2, x3)) in any add/mul argument order
bool check_rope_pointwise_module(const_module_ref pm)
{
    auto ret = std::prev(pm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return false;
    auto add = ret->inputs().front();
    if(add->name() != "add")
        return false;
    auto pidx     = param_indices(pm);
    auto mul_idxs = [&](instruction_ref mul) -> std::set<std::size_t> {
        if(mul->name() != "mul")
            return {};
        std::set<std::size_t> result;
        for(auto input : mul->inputs())
        {
            auto it = pidx.find(input);
            if(it == pidx.end())
                return {};
            result.insert(it->second);
        }
        return result;
    };
    auto s1                        = mul_idxs(add->inputs().at(0));
    auto s2                        = mul_idxs(add->inputs().at(1));
    const std::set<std::size_t> lo = {0, 1};
    const std::set<std::size_t> hi = {2, 3};
    return (s1 == lo and s2 == hi) or (s1 == hi and s2 == lo);
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
    instruction_ref inner;
    std::optional<float> scale;
    for(auto input : outer->inputs())
    {
        if(auto lit = get_scalar_literal(input))
            scale = lit;
        else
            inner = input;
    }
    if(not scale.has_value())
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

// out = mul(mul(x0, x1), x2)
bool check_norm_apply_module(const_module_ref pm)
{
    auto ret = std::prev(pm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return false;
    auto outer = ret->inputs().front();
    if(outer->name() != "mul")
        return false;
    auto pidx      = param_indices(pm);
    auto param_idx = [&](instruction_ref ins) -> std::optional<std::size_t> {
        auto it = pidx.find(ins);
        if(it == pidx.end())
            return std::nullopt;
        return it->second;
    };
    instruction_ref inner;
    bool outer_has_w = false;
    for(auto input : outer->inputs())
    {
        if(param_idx(input) == std::optional<std::size_t>{2})
            outer_has_w = true;
        else
            inner = input;
    }
    if(not outer_has_w or inner->name() != "mul")
        return false;
    return param_idx(inner->inputs().at(0)) == std::optional<std::size_t>{0} and
           param_idx(inner->inputs().at(1)) == std::optional<std::size_t>{1};
}

struct rmsnorm_info
{
    float eps      = 0;
    float ss_scale = 0;
    instruction_ref weight;
};

// Match a fused_reduce module of the form:
//   pw_square_scale(x0) -> reduce_sum[last axis] -> pw_eps_rsqrt -> multibroadcast
//   -> pw_apply(x0, rms, x1)
std::optional<rmsnorm_info> match_rmsnorm(instruction_ref fr)
{
    if(fr->name() != "fused_reduce")
        return std::nullopt;
    auto v    = fr->get_operator().to_value();
    auto axes = v.at("axes").to_vector<int64_t>();
    auto rank = static_cast<int64_t>(fr->get_shape().lens().size());
    if(axes != std::vector<int64_t>{rank - 1})
        return std::nullopt;
    if(fr->inputs().size() != 2)
        return std::nullopt;
    auto* rm = fr->module_inputs().front();

    auto ret = std::prev(rm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto apply = ret->inputs().front();
    if(apply->name() != "pointwise" or apply->inputs().size() != 3)
        return std::nullopt;
    if(not check_norm_apply_module(apply->module_inputs().front()))
        return std::nullopt;
    auto data = apply->inputs().at(0);
    auto mb   = apply->inputs().at(1);
    auto w    = apply->inputs().at(2);
    if(data->name() != "@param" or w->name() != "@param" or mb->name() != "multibroadcast")
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

    // map the module data/weight params back to the fused_reduce inputs
    auto pidx = param_indices(rm);
    if(pidx.at(data) != 0 or pidx.at(w) != 1)
        return std::nullopt;
    rmsnorm_info result;
    result.eps      = *eps;
    result.ss_scale = *scale;
    result.weight   = fr->inputs().at(1);
    return result;
}

// Walk through a broadcast to the raw 1-d weight
std::optional<instruction_ref> raw_norm_weight(instruction_ref ins, std::size_t d)
{
    while(contains({"broadcast", "multibroadcast", "reshape", "unsqueeze", "contiguous"},
                   ins->name()))
        ins = ins->inputs().front();
    if(ins->get_shape().lens() != std::vector<std::size_t>{d})
        return std::nullopt;
    if(not ins->get_shape().packed())
        return std::nullopt;
    return ins;
}

// slice(qkv) -> reshape{b,1,n,d} feeding the fused_reduce data input
std::optional<instruction_ref> qkv_of_norm_input(instruction_ref ins, int64_t start, int64_t end)
{
    if(ins->name() != "reshape" and ins->name() != "reshape_lazy")
        return std::nullopt;
    auto sl = ins->inputs().front();
    if(not is_slice(sl, {2}, {start}, {end}))
        return std::nullopt;
    return sl->inputs().front();
}

struct find_rope_qk_norm
{
    auto matcher() const
    {
        return match::name("pointwise")(match::nargs(4),
                                        match::any_of[match::inputs()](match::name("concat")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(not check_rope_pointwise_module(ins->module_inputs().front()))
            return;

        // arguments pair up as (x0*x1) + (x2*x3); identify (rotated, sin) and (qk, cos)
        auto args                                                        = ins->inputs();
        std::array<std::pair<instruction_ref, instruction_ref>, 2> pairs = {
            {{args[0], args[1]}, {args[2], args[3]}}};
        instruction_ref rot, ssin_b, qk, cos_b;
        bool found_rot = false, found_qk = false;
        for(auto [a, b] : pairs)
        {
            if(a->name() == "concat")
            {
                rot       = a;
                ssin_b    = b;
                found_rot = true;
            }
            else if(b->name() == "concat")
            {
                rot       = b;
                ssin_b    = a;
                found_rot = true;
            }
            else if(a->name() == "slice")
            {
                qk       = a;
                cos_b    = b;
                found_qk = true;
            }
            else if(b->name() == "slice")
            {
                qk       = b;
                cos_b    = a;
                found_qk = true;
            }
        }
        if(not(found_rot and found_qk))
            return;

        // shape bookkeeping from the output: {b, h, 1, d}
        auto out_lens = ins->get_shape().lens();
        if(out_lens.size() != 4 or out_lens.at(2) != 1)
            return;
        const int64_t b    = out_lens.at(0);
        const int64_t h    = out_lens.at(1);
        const int64_t d    = out_lens.at(3);
        const int64_t half = d / 2;
        if(d % 2 != 0)
            return;

        // rotate-half: concat(x[.., half:d], x[.., 0:half]) on the last axis
        if(rot->get_operator().to_value().at("axis").to<int64_t>() != 3)
            return;
        if(rot->inputs().size() != 2)
            return;
        auto s2 = rot->inputs().at(0);
        auto s1 = rot->inputs().at(1);
        if(not is_slice(s2, {1, 3}, {0, half}, {h, d}))
            return;
        if(not is_slice(s1, {1, 3}, {0, 0}, {h, half}))
            return;
        auto t = s2->inputs().front();
        if(s1->inputs().front() != t or qk->inputs().front() != t)
            return;
        if(not is_slice(qk, {1}, {0}, {h}))
            return;

        // t = transpose(reshape(concat(q_norm, k_norm, v)))
        if(t->name() != "transpose" or
           t->get_operator().to_value().at("permutation").to_vector<int64_t>() !=
               std::vector<int64_t>{0, 2, 1, 3})
            return;
        auto rsh = t->inputs().front();
        if(rsh->name() != "reshape" and rsh->name() != "reshape_lazy")
            return;
        auto cc = rsh->inputs().front();
        if(cc->name() != "concat" or cc->inputs().size() != 3 or
           cc->get_operator().to_value().at("axis").to<int64_t>() != 2)
            return;

        auto qr = cc->inputs().at(0);
        auto kr = cc->inputs().at(1);
        auto vs = cc->inputs().at(2);
        if(qr->name() != "reshape" and qr->name() != "reshape_lazy")
            return;
        if(kr->name() != "reshape" and kr->name() != "reshape_lazy")
            return;
        auto frq    = qr->inputs().front();
        auto frk    = kr->inputs().front();
        auto q_info = match_rmsnorm(frq);
        auto k_info = match_rmsnorm(frk);
        if(not q_info.has_value() or not k_info.has_value())
            return;
        if(q_info->eps != k_info->eps or q_info->ss_scale != k_info->ss_scale)
            return;

        const int64_t nq = frq->get_shape().lens().at(2);
        const int64_t nk = frk->get_shape().lens().at(2);
        if(frq->get_shape().lens().at(3) != static_cast<std::size_t>(d) or
           frk->get_shape().lens().at(3) != static_cast<std::size_t>(d))
            return;
        if(h != nq + nk)
            return;
        const int64_t total = nq + 2 * nk;

        // all three qkv slices must come from the same packed projection
        auto qkv1 = qkv_of_norm_input(frq->inputs().front(), 0, nq * d);
        auto qkv2 = qkv_of_norm_input(frk->inputs().front(), nq * d, (nq + nk) * d);
        if(not qkv1.has_value() or not qkv2.has_value())
            return;
        if(not is_slice(vs, {2}, {(nq + nk) * d}, {total * d}))
            return;
        auto qkv = *qkv1;
        if(*qkv2 != qkv or vs->inputs().front() != qkv)
            return;
        const auto& qkv_shape = qkv->get_shape();
        if(not qkv_shape.standard() or
           qkv_shape.lens() != std::vector<std::size_t>{static_cast<std::size_t>(b),
                                                        1,
                                                        static_cast<std::size_t>(total * d)})
            return;
        if(not contains({shape::bf16_type, shape::half_type, shape::float_type}, qkv_shape.type()))
            return;

        auto qw = raw_norm_weight(q_info->weight, d);
        auto kw = raw_norm_weight(k_info->weight, d);
        if(not qw.has_value() or not kw.has_value())
            return;

        // cos/sin: multibroadcast of a packed {b, 1, 1, d} tensor
        auto gathered = [&](instruction_ref bc) -> std::optional<instruction_ref> {
            if(bc->name() != "multibroadcast")
                return std::nullopt;
            auto src = bc->inputs().front();
            if(src->get_shape().elements() != static_cast<std::size_t>(b * d))
                return std::nullopt;
            if(not src->get_shape().packed())
                return std::nullopt;
            return src;
        };
        auto cos_in  = gathered(cos_b);
        auto ssin_in = gathered(ssin_b);
        if(not cos_in.has_value() or not ssin_in.has_value())
            return;

        // rewire the v heads to read straight from the qkv projection so the
        // concat becomes dead
        std::vector<instruction_ref> v_users;
        std::copy_if(t->outputs().begin(),
                     t->outputs().end(),
                     std::back_inserter(v_users),
                     [&](instruction_ref out) { return is_slice(out, {1}, {h}, {total}); });
        for(auto v_user : v_users)
        {
            auto vsl = m.insert_instruction(
                v_user,
                make_op("slice",
                        {{"axes", {2}}, {"starts", {(nq + nk) * d}}, {"ends", {total * d}}}),
                qkv);
            auto vrs = m.insert_instruction(
                v_user, make_op("reshape_lazy", {{"dims", {b, 1, nk, d}}}), vsl);
            auto vtp = m.insert_instruction(
                v_user, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), vrs);
            m.replace_instruction(v_user, vtp);
        }

        m.replace_instruction(
            ins,
            rope_qk_norm{static_cast<std::size_t>(nq), q_info->eps, q_info->ss_scale},
            {qkv, *qw, *kw, *cos_in, *ssin_in});
    }
};

} // namespace

void fuse_rope_qk_norm::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm.get_module(), find_rope_qk_norm{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
