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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.splits, "splits"));
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
    return std::min<std::size_t>(splits, 160);
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

        const auto& a_shape = a->get_shape();
        const auto& w_shape = w->get_shape();
        auto out_shape      = dot->get_shape();
        if(not a_shape.standard() or not w_shape.standard())
            return;
        if(w_shape.lens().size() != 2)
            return;
        auto k     = w_shape.lens().front();
        auto n     = w_shape.lens().back();
        auto m_dim = a_shape.elements() / a_shape.lens().back();
        if(a_shape.lens().back() != k)
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
        // overhead and the MFMA path wins; measured on qwen3-8b decode.
        if(residual == m.end())
            return;

        auto splits  = compute_splits(k, n);
        auto partial = m.insert_instruction(ins, skinny_gemm_splitk{splits}, {a, w});
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
