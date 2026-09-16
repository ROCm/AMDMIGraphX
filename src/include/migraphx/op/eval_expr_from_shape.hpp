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
#ifndef MIGRAPHX_GUARD_OPERATORS_EVAL_EXPR_FROM_SHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_EVAL_EXPR_FROM_SHAPE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/context.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/zip_view.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct eval_expr_from_shape
{
    std::vector<sym::expr> expressions{};
    std::vector<shape> input_shapes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.expressions, "expressions"), f(self.input_shapes, "input_shapes"));
    }

    std::string name() const { return "eval_expr_from_shape"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);

        std::unordered_set<sym::expr> missing;
        for(const auto& expression : expressions)
        {
            auto variables = sym::find_variables(expression);
            missing.merge(variables);
        }

        for(const auto& input : inputs)
        {
            if(not input.symbolic())
                continue;
            for(const auto& d : input.dyn_dims())
                if(d.sym_expr.name() == "variable")
                    missing.erase(sym::as_symbol(d.sym_expr));
        }
        if(not missing.empty())
            MIGRAPHX_THROW("EVAL_EXPR_FROM_SHAPE: Symbol '" + missing.begin()->to_string() +
                           "' is not a direct input dimension");

        return shape{shape::int64_type, {expressions.size()}};
    }

    void finalize(context&, const shape&, const std::vector<shape>& inputs)
    {
        input_shapes = inputs;
    }

    argument compute(const shape&, std::vector<argument> args) const
    {
        if(input_shapes.empty() or input_shapes.size() != args.size())
            MIGRAPHX_THROW("EVAL_EXPR_FROM_SHAPE: input shapes not captured; op was not finalized");

        std::unordered_map<sym::expr, std::size_t> values;
        for(auto&& [input_shape, arg] : views::zip(input_shapes, args))
        {
            const auto& lens = arg.get_shape().lens();
            if(input_shape.ndim() != lens.size())
                MIGRAPHX_THROW("EVAL_EXPR_FROM_SHAPE: Runtime input rank does not match its "
                               "symbolic shape");
            if(not input_shape.symbolic())
                continue;
            const auto& dims = input_shape.dyn_dims();
            for(auto&& [dim, len] : views::zip(dims, lens))
            {
                if(dim.sym_expr.name() != "variable")
                    continue;
                auto variable = sym::as_symbol(dim.sym_expr);
                auto result   = values.emplace(variable, len);
                if(not result.second and result.first->second != len)
                    MIGRAPHX_THROW("EVAL_EXPR_FROM_SHAPE: Repeated symbol has inconsistent runtime "
                                   "dimensions");
            }
        }

        argument result{shape{shape::int64_type, {expressions.size()}}};
        result.visit([&](auto output) {
            std::transform(expressions.begin(),
                           expressions.end(),
                           output.begin(),
                           [&](const auto& e) { return e.eval_uint(values); });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
