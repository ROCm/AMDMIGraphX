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
#include <migraphx/rewrite_appendkv.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

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
