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
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

std::optional<shape> symbolic_allocate_shape(const operation& op)
{
    if(op.name() != "allocate")
        return std::nullopt;
    auto attributes    = op.to_value();
    const auto& target = attributes.at("shape");
    if(target.is_null())
        return std::nullopt;
    auto s = from_value<shape>(target);
    if(not s.symbolic())
        return std::nullopt;
    return s;
}

struct analyze_allocate : analyzer<analyze_allocate>
{
    bool matches(const operation& op) const { return op.name() == "allocate"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        if(ins->inputs().size() != 1 or
           not symbolic_allocate_shape(ins->get_operator()).has_value())
            return symbolic_op_info{ins};
        auto result    = analyze_axes(ins, {0});
        result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source_ins,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>& values)
    {
        auto source = symbolic_allocate_shape(source_ins->get_operator());
        assert(source.has_value());
        std::vector<std::size_t> lens(source->ndim());
        std::transform(source->dyn_dims().begin(),
                       source->dyn_dims().end(),
                       lens.begin(),
                       [&](const auto& d) { return d.sym_expr.eval_uint(values); });
        std::vector<std::size_t> strides(source->ndim());
        std::transform(source->dyn_strides().begin(),
                       source->dyn_strides().end(),
                       strides.begin(),
                       [&](const auto& stride) { return stride.eval_uint(values); });
        shape target{source->type(), lens, strides};
        return m.add_instruction(make_op("allocate", {{"shape", to_value(target)}}), args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
