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
#include "migraphx/check_shapes.hpp"
#include <migraphx/rewrite_appendkv.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

struct rotary_embedding
{
    bool interleaved = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.interleaved, "interleaved"));
    }

    std::string name() const { return "rotary_embedding"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4);
        return inputs[0];
    }
};
MIGRAPHX_REGISTER_OP(rotary_embedding);


struct find_appendkv
{
    auto matcher() const { return match::name("rotary_embedding"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto val         = ins->get_operator().to_value();
        bool interleaved = val["interleaved"].to<bool>();

        auto result = op::builder::insert(
            "rotary_embedding", m, ins, ins->inputs(), {{"interleaved", interleaved}});
        m.replace_instruction(ins, result.at(0));
    }
};

struct find_ck_appendkv
{
    auto matcher() const
    {
        // K path: concat_past_present where arg(0) = slice(rotary_embedding(...)) or slice(slice(...))
        auto rotary             = match::name("rotary_embedding").bind("rotary");
        auto cur_k_with_rotary  = match::name("slice")(match::arg(0)(rotary));
        auto cur_k_without_rotary = match::name("slice")(match::arg(0)(match::name("slice")));

        auto keys = match::name("concat_past_present")(
            match::arg(0)(match::any_of(cur_k_with_rotary, cur_k_without_rotary)),
            match::arg(1)(match::any().bind("slk")),
            match::arg(2)(match::any().bind("past_k")))
            .bind("pres_k");

        // V path: concat_past_present where arg(0) = slice(transpose(...))
        auto values = match::name("concat_past_present")(
            match::arg(0)(match::name("slice")(match::arg(0)(match::name("transpose")))),
            match::arg(2)(match::any().bind("past_v")))
            .bind("pres_v");

        return match::name("group")(
            match::any_of[match::inputs()](keys),
            match::any_of[match::inputs()](values));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto group_ins = r.result;
        auto pres_k    = r.instructions["pres_k"];
        auto pres_v    = r.instructions["pres_v"];
        auto slk       = r.instructions["slk"];
        auto past_k    = r.instructions["past_k"];
        auto past_v    = r.instructions["past_v"];

        bool has_rotary = r.instructions.find("rotary") != r.instructions.end();
        auto rotary_ins = has_rotary ? r.instructions["rotary"] : instruction_ref{};

        // cur_k's input is either rotary_embedding or qk_combined (a slice)
        auto cur_k       = pres_k->inputs()[0];
        auto qk_combined = has_rotary ? rotary_ins->inputs()[0] : cur_k->inputs()[0];

        // cur_v for the CK op's vnew input
        auto cur_v = pres_v->inputs()[0];

        // TODO (task 3): insert CK op, replace instructions
        (void)m;
        (void)group_ins;
        (void)slk;
        (void)past_k;
        (void)past_v;
        (void)qk_combined;
        (void)cur_v;
    }
};

} // namespace

void rewrite_appendkv::apply(module& m) const
{
    match::find_matches(m, find_appendkv{});
    normalize_ops{}.apply(m);
    dead_code_elimination{}.apply(m);
    simplify_reshapes{.enable_gather_rewrite = true}.apply(m);
    dead_code_elimination{}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
