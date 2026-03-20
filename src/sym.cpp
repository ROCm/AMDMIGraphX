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
#include <migraphx/sym.hpp>
#include <algorithm>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace sym {

struct expr::impl
{
    node_variant node;
    std::vector<expr> children;
};

std::shared_ptr<const expr::impl> expr::make_impl(node_variant node, std::vector<expr> children)
{
    return std::make_shared<const impl>(impl{std::move(node), std::move(children)});
}

value value_min(const value& a, const value& b)
{
    return value_invoke_common([](auto x, auto y) { return x < y ? x : y; }, a, b);
}

value value_max(const value& a, const value& b)
{
    return value_invoke_common([](auto x, auto y) { return x > y ? x : y; }, a, b);
}

expr var(std::string name) { return expr(variable_node{std::move(name), {}}); }

expr var(std::string name, interval constraint)
{
    return expr(variable_node{std::move(name), {std::move(constraint)}});
}

expr arg(expr x) { return x; }

expr call_op(const op_def* op, std::vector<expr> args)
{
    return expr(op_node{op}, std::move(args));
}

expr operator+(expr ex, expr ey)
{
    return call("+", [](auto x, auto y) { return x + y; })(std::move(ex), std::move(ey));
}

expr operator-(expr ex, expr ey)
{
    return call("-", [](auto x, auto y) { return x - y; })(std::move(ex), std::move(ey));
}

expr operator*(expr ex, expr ey)
{
    return call("*", [](auto x, auto y) { return x * y; })(std::move(ex), std::move(ey));
}

expr operator/(expr ex, expr ey)
{
    return call("/", [](auto x, auto y) { return x / y; })(std::move(ex), std::move(ey));
}

expr operator-(expr e)
{
    return call("neg", [](auto x) { return -x; })(std::move(e));
}

expr sqrt(expr e)
{
    return call("sqrt", MIGRAPHX_LIFT(std::sqrt), [](interval x) -> interval {
        auto lo = std::sqrt(std::max(0.0, to<double>(x.min)));
        auto hi = std::sqrt(std::max(0.0, to<double>(x.max)));
        return {lo, hi};
    })(std::move(e));
}

value expr::eval(const std::unordered_map<std::string, value>& vars) const
{
    if(auto* n = std::get_if<literal_node>(&pimpl->node))
        return n->val;
    if(auto* n = std::get_if<variable_node>(&pimpl->node))
        return vars.at(n->name);
    auto* n = std::get_if<op_node>(&pimpl->node);
    std::vector<value> args;
    args.reserve(pimpl->children.size());
    std::transform(pimpl->children.begin(),
                   pimpl->children.end(),
                   std::back_inserter(args),
                   [&](const expr& child) { return child.eval(vars); });
    return n->op->eval(args);
}

interval expr::eval_interval(const std::unordered_map<std::string, interval>& vars) const
{
    if(auto* n = std::get_if<literal_node>(&pimpl->node))
        return {n->val, n->val};
    if(auto* n = std::get_if<variable_node>(&pimpl->node))
    {
        auto it = vars.find(n->name);
        if(it != vars.end())
            return it->second;
        if(not n->constraints.empty())
            return n->constraints.front();
        MIGRAPHX_THROW("Variable '" + n->name + "' not found in interval map");
    }
    auto* n = std::get_if<op_node>(&pimpl->node);
    std::vector<interval> args;
    args.reserve(pimpl->children.size());
    std::transform(pimpl->children.begin(),
                   pimpl->children.end(),
                   std::back_inserter(args),
                   [&](const expr& child) { return child.eval_interval(vars); });
    return n->op->eval_interval(args);
}

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
