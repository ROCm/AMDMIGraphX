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
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/sym_argument.hpp>
#include <migraphx/zip_view.hpp>
#include <algorithm>
#include <optional>
#include <unordered_map>
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

    static std::optional<sym::expr> dispatch_source(const sym::expr& expression)
    {
        if(expression.name() != "min")
            return std::nullopt;
        auto variables = sym::find_variables(expression);
        auto is_target = [](const auto& variable) {
            auto name = variable.to_string();
            return starts_with(name, "split_sym_dim_") and contains(name, "_target");
        };
        if(none_of(variables, is_target))
            return std::nullopt;
        std::vector<sym::expr> sources;
        std::copy_if(variables.begin(),
                     variables.end(),
                     std::back_inserter(sources),
                     [&](const auto& variable) { return not is_target(variable); });
        if(sources.size() != 1)
            return std::nullopt;
        return sources.front();
    }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);

        // Treat each complete symbolic dimension as a runtime value. Substitution also proves
        // expressions composed from those dimensions are resolvable.
        std::unordered_map<sym::expr, sym::expr> available;
        for(const auto& input : inputs)
        {
            if(not input.symbolic())
                continue;
            for(const auto& d : input.dyn_dims())
                if(d.is_symbolic())
                {
                    available.emplace(d.sym_expr, sym::lit(1));
                    if(d.sym_expr.name() == "variable")
                        available.emplace(sym::as_symbol(d.sym_expr), sym::lit(1));
                    auto source = dispatch_source(d.sym_expr);
                    if(source.has_value())
                        available.emplace(*source, sym::lit(1));
                }
        }
        for(const auto& expression : expressions)
        {
            auto unresolved = sym::find_variables(expression.subs(available));
            if(not unresolved.empty())
                MIGRAPHX_THROW("EVAL_EXPR_FROM_SHAPE: Expression '" + expression.to_string() +
                               "' cannot be resolved from input shapes " + to_string_range(inputs));
        }

        return shape{shape::int64_type, {expressions.size()}};
    }

    sym_argument symbolic_compute(const shape& output_shape, const std::vector<sym_argument>&) const
    {
        return {expressions, output_shape};
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
                if(not dim.is_symbolic())
                    continue;
                auto add_value = [&](const sym::expr& expression, std::size_t value) {
                    auto result = values.emplace(expression, value);
                    if(not result.second and result.first->second != value)
                        MIGRAPHX_THROW(
                            "EVAL_EXPR_FROM_SHAPE: Repeated expression has inconsistent runtime "
                            "dimensions");
                };
                add_value(dim.sym_expr, len);
                if(dim.sym_expr.name() == "variable")
                    add_value(sym::as_symbol(dim.sym_expr), len);
                auto source = dispatch_source(dim.sym_expr);
                if(source.has_value())
                    add_value(*source, len);
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
