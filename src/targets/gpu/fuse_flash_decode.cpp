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
#include <migraphx/gpu/fuse_flash_decode.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/env.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_KV_FLASH_DECODE);

namespace {

constexpr std::size_t min_seq_len = 1024;
constexpr std::size_t target_wgs  = 2048;
constexpr std::size_t min_chunk   = 512;

struct kv_flash_decode_splitk
{
    std::size_t q_heads  = 0;
    std::size_t kv_heads = 0;
    std::size_t groups   = 1;
    float scale          = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.q_heads, "q_heads"),
                    f(self.kv_heads, "kv_heads"),
                    f(self.groups, "groups"),
                    f(self.scale, "scale"));
    }

    std::string name() const { return "gpu::kv_flash_decode_splitk"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // qk, k_cache, v_cache, seqlens
        check_shapes{inputs, *this}.has(4);
        const auto& k = inputs.at(1);
        auto b        = k.lens().at(0);
        auto d        = k.lens().at(3);
        return shape{shape::float_type, {b, q_heads, groups, d + 1}};
    }
};
MIGRAPHX_REGISTER_OP(kv_flash_decode_splitk);

struct kv_flash_decode_reduce
{
    shape output_shape;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "gpu::kv_flash_decode_reduce"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return output_shape;
    }
};
MIGRAPHX_REGISTER_OP(kv_flash_decode_reduce);

instruction_ref skip_ops(instruction_ref ins, const std::unordered_set<std::string>& names)
{
    while(contains(names, ins->name()) and ins->inputs().size() == 1)
        ins = ins->inputs().front();
    return ins;
}

const std::unordered_set<std::string>& shape_ops()
{
    static const std::unordered_set<std::string> names = {
        "multibroadcast", "broadcast", "reshape", "unsqueeze", "squeeze", "transpose", "convert"};
    return names;
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

bool is_iota_literal(instruction_ref ins)
{
    if(ins->name() != "@literal")
        return false;
    if(ins->get_shape().lens().size() != 1)
        return false;
    bool result = true;
    ins->get_literal().visit([&](auto data) {
        for(std::size_t i = 0; i < data.size(); i++)
        {
            if(static_cast<std::size_t>(data[i]) != i)
            {
                result = false;
                break;
            }
        }
    });
    return result;
}

bool is_reduce_last_axis(instruction_ref ins, const std::string& name)
{
    if(ins->name() != name)
        return false;
    auto axes = ins->get_operator().to_value().at("axes").to_vector<int64_t>();
    auto rank = static_cast<int64_t>(ins->inputs().front()->get_shape().lens().size());
    return axes == std::vector<int64_t>{rank - 1};
}

struct decode_attention_info
{
    instruction_ref qk_param;
    instruction_ref k_param;
    instruction_ref v_param;
    instruction_ref sl_param;
    std::size_t q_heads  = 0;
    std::size_t kv_heads = 0;
    float scale          = 1.0f;
};

// Verify the canonical kv-cache decode attention module produced by
// find_kv_cache_attention and extract the roles of its parameters
std::optional<decode_attention_info> match_decode_attention(const_module_ref sm)
{
    decode_attention_info info;

    auto ret = std::prev(sm->end());
    if(ret->name() != "@return" or ret->inputs().size() != 1)
        return std::nullopt;
    auto rsh = ret->inputs().front();
    if(rsh->name() != "reshape")
        return std::nullopt;
    auto tp = rsh->inputs().front();
    if(tp->name() != "transpose")
        return std::nullopt;
    auto dot2 = tp->inputs().front();
    if(dot2->name() != "dot")
        return std::nullopt;

    // probabilities: convert(div(exp, broadcast(reduce_sum)))
    auto div = skip_ops(dot2->inputs().at(0), {"convert"});
    if(div->name() != "div")
        return std::nullopt;
    auto expi = div->inputs().at(0);
    if(expi->name() != "exp")
        return std::nullopt;
    auto rsum = skip_ops(div->inputs().at(1), {"multibroadcast"});
    if(not is_reduce_last_axis(rsum, "reduce_sum") or rsum->inputs().front() != expi)
        return std::nullopt;
    auto sub = expi->inputs().front();
    if(sub->name() != "sub")
        return std::nullopt;
    auto where = sub->inputs().at(0);
    if(where->name() != "where")
        return std::nullopt;
    auto rmax = skip_ops(sub->inputs().at(1), {"multibroadcast"});
    if(not is_reduce_last_axis(rmax, "reduce_max") or rmax->inputs().front() != where)
        return std::nullopt;

    // masked value must be a huge negative scalar
    auto ninf = get_scalar_literal(skip_ops(where->inputs().at(1), shape_ops()));
    if(not ninf.has_value() or *ninf > -1e30f)
        return std::nullopt;

    // mask: greater(iota, seqlens)
    auto greater = skip_ops(where->inputs().at(0), shape_ops());
    if(greater->name() != "greater")
        return std::nullopt;
    if(not is_iota_literal(skip_ops(greater->inputs().at(0), shape_ops())))
        return std::nullopt;
    auto sl = skip_ops(greater->inputs().at(1), shape_ops());
    if(sl->name() != "@param")
        return std::nullopt;
    info.sl_param = sl;

    // scores: mul(convert(dot1), scale)
    auto mul = where->inputs().at(2);
    if(mul->name() != "mul")
        return std::nullopt;
    instruction_ref dot1 = sm->end();
    for(auto arg : mul->inputs())
    {
        auto root = skip_ops(arg, {"convert"});
        if(root->name() == "dot")
        {
            dot1 = root;
        }
        else if(auto scale = get_scalar_literal(skip_ops(arg, shape_ops())))
        {
            info.scale = *scale;
        }
        else
        {
            return std::nullopt;
        }
    }
    if(dot1 == sm->end())
        return std::nullopt;

    // decode only: single query position
    if(dot1->get_shape().lens().at(2) != 1)
        return std::nullopt;

    // q: slice of the packed roped qk heads
    auto qslice = dot1->inputs().at(0);
    if(qslice->name() != "slice")
        return std::nullopt;
    auto sv = qslice->get_operator().to_value();
    if(sv.at("axes").to_vector<int64_t>() != std::vector<int64_t>{1} or
       sv.at("starts").to_vector<int64_t>() != std::vector<int64_t>{0})
        return std::nullopt;
    auto qk = qslice->inputs().front();
    if(qk->name() != "@param")
        return std::nullopt;
    info.qk_param = qk;
    info.q_heads  = sv.at("ends").to_vector<int64_t>().front();

    // k and v: the GQA-broadcast kv-cache params
    auto k = skip_ops(dot1->inputs().at(1), shape_ops());
    auto v = skip_ops(dot2->inputs().at(1), shape_ops());
    if(k->name() != "@param" or v->name() != "@param" or k == v)
        return std::nullopt;
    if(k->get_shape().lens().size() != 4 or k->get_shape() != v->get_shape())
        return std::nullopt;
    if(not k->get_shape().standard() or not v->get_shape().standard())
        return std::nullopt;
    info.k_param  = k;
    info.v_param  = v;
    info.kv_heads = k->get_shape().lens().at(1);

    auto d = k->get_shape().lens().at(3);
    if(info.kv_heads == 0 or info.q_heads % info.kv_heads != 0)
        return std::nullopt;
    if((info.q_heads / info.kv_heads) > 8)
        return std::nullopt;
    if(d % 64 != 0 or d > 256)
        return std::nullopt;
    // the split kernel computes scores with bf16 MFMA instructions
    if(k->get_shape().type() != shape::bf16_type)
        return std::nullopt;
    if(info.sl_param->get_shape().type() != shape::int32_type)
        return std::nullopt;
    return info;
}

struct find_kv_flash_decode
{
    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "kv_cache_attention"));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto group_ins = r.result;
        auto* sm       = group_ins->module_inputs().front();
        auto info      = match_decode_attention(sm);
        if(not info.has_value())
            return;

        auto n = info->k_param->get_shape().lens().at(2);
        if(n < min_seq_len)
            return;
        auto b      = info->k_param->get_shape().lens().at(0);
        auto groups = std::max<std::size_t>(target_wgs / (b * info->kv_heads), 1);
        groups      = std::min(groups, n / min_chunk);
        if(groups < 2)
            return;

        auto param_map = sm->get_ins_param_map(group_ins->inputs(), true);
        auto qk        = param_map.at(info->qk_param);
        auto k         = param_map.at(info->k_param);
        auto v         = param_map.at(info->v_param);
        auto sl        = param_map.at(info->sl_param);
        if(qk->get_shape().strides().back() != 1)
            return;

        auto splitk = m.insert_instruction(
            group_ins,
            kv_flash_decode_splitk{info->q_heads, info->kv_heads, groups, info->scale},
            {qk, k, v, sl});
        m.replace_instruction(group_ins, kv_flash_decode_reduce{group_ins->get_shape()}, {splitk});
    }
};

} // namespace

void fuse_flash_decode::apply(module_pass_manager& mpm) const
{
    if(enabled(MIGRAPHX_DISABLE_KV_FLASH_DECODE{}))
        return;
    match::find_matches(mpm.get_module(), find_kv_flash_decode{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
