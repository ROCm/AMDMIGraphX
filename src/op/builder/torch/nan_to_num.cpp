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

#include <limits>
#include <vector>
#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// nan_to_num has no native op: replace NaN/+inf/-inf with the given values.
struct torch_nan_to_num : op_builder<torch_nan_to_num>
{
    float nan    = 0.0f;
    float posinf = std::numeric_limits<float>::max();
    float neginf = std::numeric_limits<float>::lowest();

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.nan, "nan"), f(self.posinf, "posinf"), f(self.neginf, "neginf"));
    }

    static std::vector<std::string> names() { return {"tm::nan_to_num"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto type = x->get_shape().type();

        auto nan_lit    = m.add_literal({type, {nan}});
        auto zero       = m.add_literal({type, {0.0f}});
        auto posinf_lit = m.add_literal({type, {posinf}});
        auto neginf_lit = m.add_literal({type, {neginf}});

        // where selects per-element, so inputs are broadcast but not type-promoted
        const common_options no_promote{.common_type = false};
        const auto where = make_op("where");
        auto select = [&](instruction_ref cond, instruction_ref val, instruction_ref other) {
            return insert_common_op(m, ins, where, {cond, val, other}, no_promote);
        };

        auto is_nan   = m.insert_instruction(ins, make_op("isnan"), x);
        auto result   = select(is_nan, nan_lit, x);
        auto is_inf   = m.insert_instruction(ins, make_op("isinf"), x);
        auto less     = insert_common_op(m, ins, "less", x, zero);
        auto greater  = insert_common_op(m, ins, "greater", x, zero);
        auto neg_mask = insert_common_op(m, ins, "logical_and", less, is_inf);
        auto pos_mask = insert_common_op(m, ins, "logical_and", greater, is_inf);
        result        = select(neg_mask, neginf_lit, result);
        return {select(pos_mask, posinf_lit, result)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
