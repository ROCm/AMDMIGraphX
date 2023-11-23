/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/permutation.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
#include <migraphx/gpu/ck.hpp>
#endif
#ifdef MIGRAPHX_MLIR
#include <migraphx/gpu/fuse_mlir.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

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

template <auto CheckDot>
struct find_gemm_softmax_gemm
{
    auto matcher() const
    {
        auto gemm1 =
            match::skip(match::name("contiguous"))(match::name("dot")(CheckDot().bind("gemm1")));
        auto mul = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
        auto softmax = match::name("softmax")(match::arg(0)(mul)).bind("softmax");

        return match::name("dot")(CheckDot().bind("gemm2"))(match::arg(0)(softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];
        auto scale_lit = r.instructions["scale"];

        float scale = 1.0;
        scale_lit->eval().visit([&](const auto s) {
            // CK only supports single-valued scale
            if(std::all_of(
                   s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                scale = s.front();
            else
                return;
        });

        auto inputs = gemm1_ins->inputs();            // A, B
        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, pre_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL

auto is_ck_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "dot")
            r eturn false;
        if(not pre_gemm_softmax_gemm::is_ck_supported_type(ins->get_shape().type()))
            return false;
        return true;
    });
}

#endif

#ifdef MIGRAPHX_MLIR

auto is_mlir_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref gemm) {
        if(std::any_of(gemm->inputs().begin(), gemm->inputs().end(), [&](auto i) {
               return not contains(
                   {shape::type_t::float_type, shape::type_t::half_type, shape::type_t::int8_type},
                   i->get_shape().type());
           }))
        {
            return false;
        }
        return true;
    });
}

#endif

} // namespace

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm.get_module(), find_layernorm{});
    mpm.run_pass(dead_code_elimination{});
    match::find_matches(mpm.get_module(), find_add_layernorm{});
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
    if(enabled(MIGRAPHX_ENABLE_CK{}))
        match::find_matches(mpm, find_gemm_softmax_gemm<is_ck_gemm>{});
#endif
#ifdef MIGRAPHX_MLIR
    if(mlir_attention_enabled())
    {
        match::find_matches(mpm, find_gemm_softmax_gemm<is_mlir_gemm>{});
    }
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
