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
#include <migraphx/match/layernorm.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace {

template <class Derived, std::size_t N>
struct layernorm_base
{
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
        if(s.scalar())
        {
            return s;
        }
        else if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
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

        m.replace_instruction(ins, layernorm{}, x_ins);
    }
};

struct find_add_layernorm
{
    auto matcher() const
    {
        return match::layernorm()(match::var("x")(match::name("add").bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];

        m.replace_instruction(ins, add_layernorm{}, add_ins->inputs());
    }
};

struct find_gpulayernorm
{
    auto matcher() const { return match::layernorm(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];

        if(not x_ins->get_shape().standard())
            x_ins = m.insert_instruction(ins, make_op("contiguous"), x_ins);

        auto relements = x_ins->get_shape().lens().back();

        if(relements > 1024 or (relements % 4 != 0 and relements > 256))
            return;

        auto a = m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(x_ins->get_shape())}}));
        m.replace_instruction(ins, make_op("gpu::layernorm"), x_ins, a);
    }
};

struct find_gputriaddlayernorm
{
    auto matcher() const
    {
        auto add1 =
            match::name("add")(match::none_of(match::is_constant()),
                               match::args(match::any().bind("z1"), match::any().bind("z2")));
        auto add2 = match::name("add")(match::either_arg(0, 1)(add1, match::any().bind("z3")));
        return match::layernorm()(match::var("x")(add2));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["z1"];
        auto y_ins = r.instructions["z2"];
        auto z_ins = r.instructions["z3"];

        for(auto* pins : {&x_ins, &y_ins, &z_ins})
        {
            if(not(*pins)->get_shape().standard())
                *pins = m.insert_instruction(ins, make_op("contiguous"), *pins);
        }

        auto relements = x_ins->get_shape().lens().back();

        if(relements > 1024 or (relements % 4 != 0 and relements > 256))
            return;

        auto a = m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(x_ins->get_shape())}}));
        m.replace_instruction(ins, make_op("gpu::triadd_layernorm"), x_ins, y_ins, z_ins, a);
    }
};
} // namespace

void prefuse_ops::apply(module& m) const
{
    match::find_matches(m, find_add_layernorm{}, find_layernorm{});
    // match::find_matches(m, find_gputriaddlayernorm{}, find_gpulayernorm{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
