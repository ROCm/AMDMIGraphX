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
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

std::optional<shape::dynamic_dimension> symbolic_range_dim(const operation& op)
{
    if(op.name() != "dynamic_range")
        return std::nullopt;
    auto attributes = op.to_value();
    if(not attributes.contains("output_dim") or attributes.at("output_dim").is_null())
        return std::nullopt;
    auto output_dim = from_value<shape::dynamic_dimension>(attributes.at("output_dim"));
    if(not output_dim.is_symbolic())
        return std::nullopt;
    return output_dim;
}

struct analyze_dynamic_range : analyzer<analyze_dynamic_range>
{
    bool matches(const operation& op) const { return op.name() == "dynamic_range"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        const auto& op = ins->get_operator();
        auto inputs    = to_shapes(ins->inputs());
        if(inputs.size() != 3 or inputs.front().type() != shape::int64_type or
           not symbolic_range_dim(op).has_value())
            return symbolic_op_info{ins};
        auto result    = analyze_axes(ins);
        result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>& values)
    {
        assert(args.size() == 3);
        auto output_dim = symbolic_range_dim(source->get_operator());
        assert(output_dim.has_value());
        auto length = output_dim->sym_expr.eval_uint(values);
        std::vector<int64_t> indices(length);
        std::iota(indices.begin(), indices.end(), int64_t{0});
        auto index = m.add_literal(literal{shape{shape::int64_type, {length}}, indices});
        auto start =
            m.add_instruction(make_op("multibroadcast", {{"out_lens", {length}}}), args.front());
        auto delta =
            m.add_instruction(make_op("multibroadcast", {{"out_lens", {length}}}), args.back());
        auto scaled = m.add_instruction(make_op("mul"), index, delta);
        return m.add_instruction(make_op("add"), start, scaled);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
