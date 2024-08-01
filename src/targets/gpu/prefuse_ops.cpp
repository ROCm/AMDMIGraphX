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
 */
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/group_query_attention.hpp>
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
#include <migraphx/gpu/ck.hpp>
#endif
#include <migraphx/gpu/fuse_mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_LAYERNORM_FUSION);

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

auto is_mlir_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(not mlir_attention_enabled())
            return false;
        if(ins->name() != "dot")
            return false;
        return std::all_of(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
            return pre_gemm_softmax_gemm::is_mlir_supported_type(i->get_shape().type());
        });
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
            match::any_of(is_ck_gemm(), is_mlir_gemm(), is_test_gemm(enable_attention))
                .bind("gemm1")));
        auto mul   = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
        auto add = match::name("add")(is_bias_supported(),
                                      match::nargs(2),
                                      match::any_arg(0, 1)(match::none_of(mul).bind("bias")));
        auto softmax =
            match::name("softmax")(match::arg(0)(match::any_of(mul, add, gemm1))).bind("softmax");

        return match::name("dot")(
            match::any_of(is_ck_gemm(), is_mlir_gemm(), is_test_gemm(enable_attention))
                .bind("gemm2"))(match::arg(0)(softmax));
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
        if(contains(r.instructions, "bias"))
        {
            inputs.push_back(r.instructions["bias"]);
        }
        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, pre_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

struct gpu_group_query_attention : op::group_query_attention
{
    std::string name() const { return "gpu::group_query_attention"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // std::cout << "new compute shape : " << inputs.size() << std::endl;
        if (inputs.size() == 8)
        {
            auto query_lens = inputs.front().lens();
            std::size_t q_hidden_size = (query_lens[1] * query_lens[3] * num_heads) / (num_heads + 2 * kv_num_heads);
            std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(2), q_hidden_size};
            shape output_shape{inputs.front().type(), output_lens};
            return shape({output_shape, inputs[1], inputs[2]});
        }
        auto query_lens = inputs.front().lens();
        std::size_t q_hidden_size = (query_lens[1] * query_lens[3] * num_heads) / (num_heads + 2 * kv_num_heads);
        std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(2), q_hidden_size};
        shape output_shape{inputs.front().type(), output_lens};
        return shape({output_shape, inputs[1], inputs[2]});
    }
};
MIGRAPHX_REGISTER_OP(gpu_group_query_attention);

struct find_group_query_attention
{
    auto matcher() const
    {
        return match::name("group_query_attention");
    }

    // instruction_ref insert_allocation(module mod, instruction_ref ins, const shape& s) const
    // {
    //     return mod->insert_instruction(ins, make_op("allocate", {{"shape", to_value(s)}}));
    // }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto inputs = ins->inputs();

        auto v = ins->get_operator().to_value();
        assert(v.contains("num_heads"));
        auto num_heads = v.at("num_heads").to<int>();
        assert(v.contains("kv_num_heads"));
        auto kv_num_heads = v.at("kv_num_heads").to<int>();
        auto s      = ins->get_shape();
        // auto output = insert_allocation(mpm.get_module(), ins, s);
        // auto inputs = ins->inputs();
        // inputs.push_back(output);

        auto q_shape              = inputs[0]->get_shape();
        auto q_lens               = q_shape.lens();
        auto input_type = q_shape.type();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        auto past_key_shape       = inputs[3]->get_shape();
        auto past_key_lens        = past_key_shape.lens();
        auto past_sequence_length = past_key_lens[2];
        std::size_t q_hidden_size = q_lens[2];
        std::size_t head_size     = q_hidden_size / (num_heads + 2 * kv_num_heads);
        q_hidden_size = head_size * num_heads; 
        const bool packed_qkv = true;

        std::size_t rotary_dim = inputs[7]->get_shape().lens()[1] * 2;
        std::size_t present_kv_seqlen = 4096;

        shape output_shape{input_type, {batch_size, sequence_length, q_hidden_size}};
        shape qkv_rotary_shape{input_type, {batch_size,
                                static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                sequence_length,
                                head_size}};
        shape kv_shape{input_type, 
                            {batch_size,
                                static_cast<std::size_t>(kv_num_heads),
                                past_sequence_length,
                                head_size}};
        shape attn_probs_shape{input_type, {batch_size, static_cast<std::size_t>(num_heads), sequence_length, present_kv_seqlen}};
        // std::cout << "mid" << std::endl;
        // insert transpose on qkv
        std::vector<std::size_t> bsnh{batch_size,
                                        sequence_length,
                                        static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                        head_size};
        // auto transposed_qkv = insert_allocation(inputs.at(0), inputs.at(0)->get_shape());                                            
        auto transposed_qkv = mpm.get_module().insert_instruction(ins, make_op("reshape", {{"dims", bsnh}}), inputs.at(0));
        transposed_qkv = mpm.get_module().insert_instruction(ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);
        transposed_qkv = mpm.get_module().insert_instruction(ins, make_op("contiguous"), transposed_qkv);
        // insert qkv_rotary with qkv_rotary_shape
        // auto qkv_rotary = insert_allocation(mpm.get_module(), ins, qkv_rotary_shape);
        // // insert attn_probs with attn_probs_shape
        // auto attn_probs = insert_allocation(mpm.get_module(), ins, attn_probs_shape);
        // std::cout << "mid2" << std::endl;
        std::vector<instruction_ref> new_inputs{
                                        transposed_qkv, 
                                        inputs.at(3), 
                                        inputs.at(4),
                                        inputs.at(5),
                                        inputs.at(7),
                                        inputs.at(8)};
        // std::cout << "mid3" << std::endl;
        // auto ret = mod->replace_instruction(
        //     ins,
        //     make_op("gpu::precompile_op", {{"op", to_value(ins->get_operator())}}),
        //     new_inputs);




        mpm.get_module().replace_instruction(
            ins, gpu_group_query_attention{v.at("do_rotary").to<int>(),
                                           v.at("kv_num_heads").to<int>(),
                                           v.at("local_window_size").to<int>(),
                                           v.at("num_heads").to<std::size_t>(),
                                           v.at("rotary_interleaved").to<int>(),
                                           v.at("scale").to<float>()}, new_inputs);
        // mpm.get_module().debug_print();
    }
};

} // namespace

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    if(not enabled(MIGRAPHX_DISABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    match::find_matches(mpm, find_group_query_attention{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
