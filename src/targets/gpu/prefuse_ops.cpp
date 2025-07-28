/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
#include <migraphx/gpu/ck.hpp>
#endif
#include <migraphx/gpu/fuse_mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_LAYERNORM_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR);

namespace {

template <class Derived, std::size_t N>
struct layernorm_base
{
    float epsilon = 1e-12f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
    {
        std::size_t nargs = N;
        if(not mods.empty())
        {
            auto* pm = mods.front();
            nargs += pm->get_parameter_names().size() - 1;
        }
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(nargs);
        auto s = inputs.front();
        auto t = s.type();
        if(not mods.empty())
            t = mods.front()->get_output_shapes().front().type();

        // Scalar output if all inputs are scalar
        if(inputs.front().elements() == 1 and
           all_of(inputs, [](const auto& ss) { return ss.scalar(); }))
            return inputs.front();
        auto l_s = shape::from_permutation(
            t, s.lens(), find_permutation(std::vector<shape>(inputs.begin(), inputs.begin() + N)));
        // just prelayernorm or preadd_layernorm
        if(nargs <= N)
            return l_s;
        // else, layernorm + pointwise fusion, preserve layout of fused op
        std::vector<shape> lp_s(inputs.begin() + N, inputs.end());
        lp_s.insert(lp_s.begin(), l_s);
        return shape::from_permutation(t, s.lens(), find_permutation(lp_s));
    }
};

struct layernorm : layernorm_base<layernorm, 1>
{

    std::string name() const { return "gpu::prelayernorm"; }
};
MIGRAPHX_REGISTER_OP(layernorm);

struct add_layernorm : layernorm_base<add_layernorm, 2>
{
    std::string name() const { return "gpu::preadd_layernorm"; }
};
MIGRAPHX_REGISTER_OP(add_layernorm);

struct find_layernorm
{
    auto matcher() const { return match::layernorm(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        float eps  = 0;
        if(contains(r.instructions, "eps"))
            eps = r.instructions["eps"]->eval().at<float>();

        m.replace_instruction(ins, layernorm{eps}, x_ins);
    }
};

struct find_add_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::prelayernorm")(
            match::args(match::name("add")(match::used_once()).bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];
        auto op      = any_cast<layernorm>(ins->get_operator());

        m.replace_instruction(ins, add_layernorm{op.epsilon}, add_ins->inputs());
    }
};

struct pre_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::pre_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(pre_gemm_softmax_gemm);

auto is_ck_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        if(not enabled(MIGRAPHX_ENABLE_CK{}))
            return false;
        if(ins->name() != "dot")
            return false;
        if(not pre_gemm_softmax_gemm::is_ck_supported_type(ins->get_shape().type()))
            return false;
        return true;
#else
        (void)ins;
        return false;
#endif
    });
}

auto is_test_gemm(bool enable_attention)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "dot")
            return false;
        return enable_attention;
    });
}

auto is_bias_supported()
{
    return match::make_basic_pred_matcher([=](instruction_ref) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        return not enabled(MIGRAPHX_ENABLE_CK{});
#else
        return true;
#endif
    });
}

struct find_gemm_softmax_gemm
{
    bool enable_attention = false;

    auto matcher() const
    {
        auto gemm1 = match::skip(match::name("contiguous"))(match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm1")));
        auto mul   = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
        auto where = match::name("where")(match::arg(2)(match::is_constant().bind("select_const")),
                                          match::arg(1)(mul),
                                          match::arg(0)(match::any().bind("select_cond")));
        auto add =
            match::name("add")(is_bias_supported(),
                               match::nargs(2),
                               match::either_arg(0, 1)(match::none_of(mul).bind("bias"), mul));
        auto softmax = match::name("softmax")(match::arg(0)(match::any_of(mul, add, gemm1, where)))
                           .bind("softmax");

        return match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm2"))(
            match::arg(0)(softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];

        float scale = 1.0;
        if(contains(r.instructions, "scale"))
        {
            auto scale_lit = r.instructions["scale"];
            // CK only supports single-valued scale
            scale_lit->eval().visit([&](const auto s) {
                // CK only supports single-valued scale
                if(not std::all_of(
                       s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                    return;
                scale = s.front();
            });
        }

        auto inputs = gemm1_ins->inputs(); // A, B
        if(contains(r.instructions, "select_cond"))
        {
            inputs.push_back(r.instructions["select_cond"]);
            inputs.push_back(r.instructions["select_const"]);
        }
        if(contains(r.instructions, "bias"))
        {
            inputs.push_back(r.instructions["bias"]);
        }

        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, pre_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

struct base_group_query_attention
{
    bool do_rotary           = false;
    std::size_t kv_num_heads = 0;
    int local_window_size    = -1;
    std::size_t num_heads    = 1;
    bool rotary_interleaved  = false;
    float scale              = 1.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.do_rotary, "do_rotary"),
                    f(self.kv_num_heads, "kv_num_heads"),
                    f(self.local_window_size, "local_window_size"),
                    f(self.num_heads, "num_heads"),
                    f(self.rotary_interleaved, "rotary_interleaved"),
                    f(self.scale, "scale"));
    }
};

struct gpu_gqa_rotary_embedding : base_group_query_attention
{
    std::string name() const { return "gpu::gqa_rotary_embedding"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
};
MIGRAPHX_REGISTER_OP(gpu_gqa_rotary_embedding);

struct gpu_concat_past_present : base_group_query_attention
{
    std::string name() const { return "gpu::concat_past_present"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.back(); }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};
MIGRAPHX_REGISTER_OP(gpu_concat_past_present);

struct find_group_query_attention
{
    std::size_t* counter = nullptr;

    auto matcher() const { return match::name("group_query_attention"); }

    auto finalize_attention_module(module_ref m) const
    {
        eliminate_common_subexpression{}.apply(*m);
        dead_code_elimination{}.apply(*m);
    }

    std::string get_count() const
    {
        if(counter == nullptr)
            MIGRAPHX_THROW("Invalid counter");
        return std::to_string((*counter)++);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto inputs = ins->inputs();
        auto val    = ins->get_operator().to_value();

        auto num_heads          = val.at("num_heads").to<std::size_t>();
        auto kv_num_heads       = val.at("kv_num_heads").to<std::size_t>();
        auto do_rotary          = val.at("do_rotary").to<bool>();
        auto local_window_size  = val.at("local_window_size").to<int>();
        auto rotary_interleaved = val.at("rotary_interleaved").to<bool>();
        auto scale              = val.at("scale").to<float>();

        auto q_shape                      = inputs[0]->get_shape();
        const auto& q_lens                = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);

        std::vector<std::size_t> bsnh{
            batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size};

        auto transposed_qkv = mpm.get_module().insert_instruction(
            ins, make_op("reshape", {{"dims", bsnh}}), inputs.at(0));

        transposed_qkv = mpm.get_module().insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);

        auto rotary_qkv = transposed_qkv;
        if(do_rotary)
        {
            std::vector<instruction_ref> rotary_inputs{
                transposed_qkv, inputs.at(5), inputs.at(7), inputs.at(8)};
            rotary_qkv =
                mpm.get_module().insert_instruction(ins,
                                                    gpu_gqa_rotary_embedding{do_rotary,
                                                                             kv_num_heads,
                                                                             local_window_size,
                                                                             num_heads,
                                                                             rotary_interleaved,
                                                                             scale},
                                                    rotary_inputs);
        }

        auto pres_k   = inputs.at(3);
        auto pres_v   = inputs.at(4);
        auto slk      = inputs.at(5);
        auto rotary_k = mpm.get_module().insert_instruction(
            ins,
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            rotary_qkv);
        auto rotary_v = mpm.get_module().insert_instruction(
            ins,
            make_op("slice",
                    {{"axes", {1}},
                     {"starts", {num_heads + kv_num_heads}},
                     {"ends", {num_heads + (2 * kv_num_heads)}}}),
            rotary_qkv);
        std::vector<instruction_ref> concat_k_inputs{rotary_k, slk, pres_k};
        std::vector<instruction_ref> concat_v_inputs{rotary_v, slk, pres_v};

        pres_k = mpm.get_module().insert_instruction(
            ins,
            gpu_concat_past_present{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            concat_k_inputs);
        pres_v = mpm.get_module().insert_instruction(
            ins,
            gpu_concat_past_present{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            concat_v_inputs);

        // Adding 1 to seq_lens_k, aka past_seq_lens, to allow range literals to start at 0.
        // Putting the add inside the mlir module currently causes an error on their side,
        // so we're leaving it here until that can be solved.
        auto past_sl = mpm.get_module().insert_instruction(
            ins, make_op("convert", {{"target_type", shape::int32_type}}), inputs.at(5));
        auto one_lit = mpm.get_module().insert_literal(
            ins, literal{shape{past_sl->get_shape().type(), {1}}, {1}});
        one_lit = mpm.get_module().insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", past_sl->get_shape().lens()}}), one_lit);
        auto total_sl = mpm.get_module().insert_instruction(ins, make_op("add"), past_sl, one_lit);

        auto get_tuple_elm_0 = std::next(ins);
        auto get_tuple_elm_1 = std::next(get_tuple_elm_0);
        auto get_tuple_elm_2 = std::next(get_tuple_elm_1);

        mpm.get_module().replace_instruction(get_tuple_elm_2, pres_v);
        mpm.get_module().replace_instruction(get_tuple_elm_1, pres_k);

        auto kv_num_heads_factor = num_heads / kv_num_heads;
        auto max_seq_len         = pres_k->get_shape().lens()[2];
        total_sl                 = mpm.get_module().insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), total_sl);
        std::vector<instruction_ref> new_inputs{rotary_qkv, pres_k, pres_v, total_sl};

        module m_attn;
        std::vector<instruction_ref> attn_inputs = {rotary_qkv, pres_k, pres_v, total_sl};
        std::unordered_map<instruction_ref, instruction_ref> map_main_to_mattn;
        m_attn.add_params(attn_inputs, &map_main_to_mattn);

        auto q = m_attn.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}),
            map_main_to_mattn.at(rotary_qkv));
        auto k = map_main_to_mattn.at(pres_k);
        auto v = map_main_to_mattn.at(pres_v);
        if(kv_num_heads_factor != 1)
        {
            auto kv_new_lens  = k->get_shape().lens();
            kv_new_lens.at(1) = num_heads;
            k                 = m_attn.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
            v                 = m_attn.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
            auto kv_unsqueezed_lens  = k->get_shape().lens();
            kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
            k                        = m_attn.add_instruction(
                make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), k);
            v = m_attn.add_instruction(
                make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), v);
            k = m_attn.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), k);
            v = m_attn.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), v);
        }
        auto kt = m_attn.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto gemm1 = m_attn.add_instruction(make_op("dot"), q, kt);

        std::vector<int> range_vec(max_seq_len);
        std::iota(range_vec.begin(), range_vec.end(), 0);
        shape range_s{total_sl->get_shape().type(), {max_seq_len}};
        auto range = m_attn.add_literal(range_s, range_vec);
        std::vector<std::size_t> bnsm{batch_size, num_heads, sequence_length, max_seq_len};
        auto bc_range =
            m_attn.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), range);

        auto scalar_s = shape{rotary_qkv->get_shape().type(), {1}};
        auto ninf =
            m_attn.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
        ninf = m_attn.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

        if(float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = m_attn.add_literal(literal{scalar_s, {scale}});
        scale_ins =
            m_attn.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
        auto mul = m_attn.add_instruction(make_op("mul"), gemm1, scale_ins);

        if(sequence_length > 1)
        {
            std::vector<int> seq_range_vec(sequence_length);
            std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
            shape seq_range_s{total_sl->get_shape().type(), {sequence_length}};
            auto seq_range = m_attn.add_literal(seq_range_s, seq_range_vec);
            seq_range = m_attn.add_instruction(make_op("reshape", {{"dims", {sequence_length, 1}}}),
                                               seq_range);
            seq_range =
                m_attn.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
            auto causal_mask = m_attn.add_instruction(make_op("greater"), bc_range, seq_range);
            causal_mask      = m_attn.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
            mul = m_attn.add_instruction(make_op("where"), causal_mask, ninf, mul);
        }

        auto bc_total_sl =
            m_attn.add_instruction(make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}),
                                   map_main_to_mattn.at(total_sl));
        auto mask_comp =
            m_attn.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), bc_total_sl);
        auto mask = m_attn.add_instruction(make_op("greater"), bc_range, mask_comp);
        mask =
            m_attn.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
        auto where   = m_attn.add_instruction(make_op("where"), mask, ninf, mul);
        auto softmax = m_attn.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = m_attn.add_instruction(make_op("dot"), softmax, v);
        auto out =
            m_attn.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = m_attn.add_instruction(
            make_op("reshape", {{"dims", get_tuple_elm_0->get_shape().lens()}}), out);
        m_attn.add_return({out});

        finalize_attention_module(&m_attn);
        module_ref mpm_attn = mpm.create_module("mlir_attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        auto group_op = mpm.get_module().insert_instruction(
            ins, make_op("group", {{"tag", "attention"}}), new_inputs, {mpm_attn});
        mpm.get_module().replace_instruction(get_tuple_elm_0, group_op);
    }
};

void inline_group_sub_module(module_pass_manager& mpm)
{
    auto& m = mpm.get_module();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "group")
            continue;

        const auto& mod_inputs = ins->module_inputs();
        auto inline_mod        = m.insert_inline(ins, *mod_inputs.at(0), ins->inputs());
        m.replace_instruction(ins, inline_mod.at(0));
    }
}

} // namespace

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    if(enabled(MIGRAPHX_ENABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    match::find_matches(mpm, find_group_query_attention{.counter = &counter});

    if(enabled(MIGRAPHX_DISABLE_MLIR{}))
    {
        inline_group_sub_module(mpm);
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
