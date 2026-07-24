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
#ifndef MIGRAPHX_GUARD_OPERATORS_EVAL_EXPR_HPP
#define MIGRAPHX_GUARD_OPERATORS_EVAL_EXPR_HPP

#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct eval_expr
{
    std::vector<sym::expr> expressions{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.expressions, "expressions"));
    }

    std::string name() const { return "eval_expr"; }

    static void collect_variables(const sym::expr& e, std::vector<sym::expr>& variables)
    {
        if(e.name() == "variable")
        {
            auto variable = sym::as_symbol(e);
            if(std::none_of(variables.begin(), variables.end(), [&](const auto& v) {
                   return sym::same_symbol(v, variable);
               }))
                variables.push_back(std::move(variable));
            return;
        }
        for(const auto& child : e.children())
            collect_variables(child, variables);
    }

    static std::vector<sym::expr> direct_variables(const shape& s)
    {
        std::vector<sym::expr> result;
        if(not s.symbolic())
            return result;
        for(const auto& d : s.dyn_dims())
        {
            if(d.sym_expr.name() != "variable")
                continue;
            auto variable = sym::as_symbol(d.sym_expr);
            if(std::none_of(result.begin(), result.end(), [&](const auto& v) {
                   return sym::same_symbol(v, variable);
               }))
                result.push_back(std::move(variable));
        }
        return result;
    }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        std::vector<sym::expr> required;
        for(const auto& expression : expressions)
            collect_variables(expression, required);
        auto available = direct_variables(inputs.front());
        auto missing   = std::find_if(required.begin(), required.end(), [&](const auto& variable) {
            return std::none_of(available.begin(), available.end(), [&](const auto& v) {
                return sym::same_symbol(v, variable);
            });
        });
        if(missing != required.end())
            MIGRAPHX_THROW("EVAL_EXPR: Symbol '" + missing->to_string() +
                           "' is not a direct input dimension");
        return shape{shape::int64_type, {expressions.size()}};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        assert(args.size() == 1);
        assert(dyn_out.input_shapes.size() == 1);
        const auto& input_shape = dyn_out.input_shapes.front();
        auto lens               = args.front().get_shape().lens();
        if(input_shape.ndim() != lens.size())
            MIGRAPHX_THROW("EVAL_EXPR: Runtime input rank does not match its symbolic shape");

        std::unordered_map<sym::expr, std::size_t> values;
        if(input_shape.symbolic())
        {
            const auto& dims = input_shape.dyn_dims();
            for(std::size_t axis = 0; axis < dims.size(); ++axis)
            {
                if(dims[axis].sym_expr.name() != "variable")
                    continue;
                auto variable = sym::as_symbol(dims[axis].sym_expr);
                auto result   = values.emplace(variable, lens[axis]);
                if(not result.second and result.first->second != lens[axis])
                    MIGRAPHX_THROW(
                        "EVAL_EXPR: Repeated symbol has inconsistent runtime dimensions");
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
