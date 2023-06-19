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
#include <migraphx/gpu/prefuse_ops.hpp>
#if !defined(_MSC_VER)
#include <migraphx/match/layernorm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#endif
#include <migraphx/pass_manager.hpp>
#if !defined(_MSC_VER)
#include <migraphx/dead_code_elimination.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#if !defined(_MSC_VER)
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
        std::size_t nargs = 1;
        if(not mods.empty())
        {
            auto* pm = mods.front();
            nargs    = pm->get_parameter_names().size();
        }
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(nargs + N);
        auto s = inputs.at(0);
        auto t = s.type();
        if(not mods.empty())
            t = mods.front()->get_output_shapes().front().type();
        if(s.scalar())
        {
            return s;
        }
        else if(s.broadcasted())
        {
            return {t, s.lens()};
        }
        else
        {
            return s.with_lens(t, s.lens());
        }
    }
};

struct layernorm : layernorm_base<layernorm, 0>
{

    std::string name() const { return "gpu::prelayernorm"; }
};
MIGRAPHX_REGISTER_OP(layernorm);

struct add_layernorm : layernorm_base<add_layernorm, 1>
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
} // namespace
#endif

void prefuse_ops::apply(module_pass_manager& mpm) const
{
#if !defined(_MSC_VER)
    match::find_matches(mpm.get_module(), find_layernorm{});
    mpm.run_pass(dead_code_elimination{});
    match::find_matches(mpm.get_module(), find_add_layernorm{});
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
