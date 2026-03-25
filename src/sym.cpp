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
#include <numeric>
#include <sstream>

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

interval operator+(interval a, interval b)
{
    auto f = [](auto x, auto y) { return x + y; };
    return {value_invoke_common(f, a.min, b.min), value_invoke_common(f, a.max, b.max)};
}

interval operator-(interval a, interval b)
{
    auto f = [](auto x, auto y) { return x - y; };
    return {value_invoke_common(f, a.min, b.max), value_invoke_common(f, a.max, b.min)};
}

interval operator*(interval a, interval b)
{
    auto f  = [](auto x, auto y) { return x * y; };
    auto p1 = value_invoke_common(f, a.min, b.min);
    auto p2 = value_invoke_common(f, a.min, b.max);
    auto p3 = value_invoke_common(f, a.max, b.min);
    auto p4 = value_invoke_common(f, a.max, b.max);
    return {value_min(value_min(p1, p2), value_min(p3, p4)),
            value_max(value_max(p1, p2), value_max(p3, p4))};
}

interval operator/(interval a, interval b)
{
    auto f  = [](auto x, auto y) { return x / y; };
    auto p1 = value_invoke_common(f, a.min, b.min);
    auto p2 = value_invoke_common(f, a.min, b.max);
    auto p3 = value_invoke_common(f, a.max, b.min);
    auto p4 = value_invoke_common(f, a.max, b.max);
    return {value_min(value_min(p1, p2), value_min(p3, p4)),
            value_max(value_max(p1, p2), value_max(p3, p4))};
}

interval operator-(interval a)
{
    auto f = [](auto x) { return -x; };
    return {value_invoke_common(f, a.max), value_invoke_common(f, a.min)};
}

bool operator==(const interval& a, const interval& b) { return a.min == b.min and a.max == b.max; }

bool operator!=(const interval& a, const interval& b) { return not(a == b); }

namespace {
bool value_less(const value& a, const value& b)
{
    auto f = [](auto x, auto y) -> int64_t { return x < y ? 1 : 0; };
    return std::get<int64_t>(value_invoke_common(f, a, b)) != 0;
}
} // namespace

bool operator<(interval a, interval b) { return value_less(a.max, b.min); }

bool operator<=(interval a, interval b) { return not value_less(b.min, a.max); }

bool operator>(interval a, interval b) { return value_less(b.max, a.min); }

bool operator>=(interval a, interval b) { return not value_less(a.min, b.max); }

interval sin(interval x)
{
    double lo       = to<double>(x.min);
    double hi       = to<double>(x.max);
    const double pi = std::acos(-1.0);
    if(hi - lo >= 2.0 * pi)
        return {-1.0, 1.0};
    double slo  = std::sin(lo);
    double shi  = std::sin(hi);
    double rmin = std::min(slo, shi);
    double rmax = std::max(slo, shi);
    double k    = std::ceil((lo - pi / 2.0) / (2.0 * pi));
    if(pi / 2.0 + k * 2.0 * pi <= hi)
        rmax = 1.0;
    k = std::ceil((lo + pi / 2.0) / (2.0 * pi));
    if(-pi / 2.0 + k * 2.0 * pi <= hi)
        rmin = -1.0;
    return {rmin, rmax};
}

interval cos(interval x)
{
    double lo       = to<double>(x.min);
    double hi       = to<double>(x.max);
    const double pi = std::acos(-1.0);
    if(hi - lo >= 2.0 * pi)
        return {-1.0, 1.0};
    double clo  = std::cos(lo);
    double chi  = std::cos(hi);
    double rmin = std::min(clo, chi);
    double rmax = std::max(clo, chi);
    double k    = std::ceil(lo / (2.0 * pi));
    if(k * 2.0 * pi <= hi)
        rmax = 1.0;
    k = std::ceil((lo - pi) / (2.0 * pi));
    if(pi + k * 2.0 * pi <= hi)
        rmin = -1.0;
    return {rmin, rmax};
}

interval tan(interval x) { return {std::tan(to<double>(x.min)), std::tan(to<double>(x.max))}; }

interval exp(interval x) { return {std::exp(to<double>(x.min)), std::exp(to<double>(x.max))}; }

interval log(interval x) { return {std::log(to<double>(x.min)), std::log(to<double>(x.max))}; }

interval sqrt(interval x)
{
    auto lo = std::sqrt(std::max(0.0, to<double>(x.min)));
    auto hi = std::sqrt(std::max(0.0, to<double>(x.max)));
    return {lo, hi};
}

interval abs(interval x)
{
    double lo = to<double>(x.min);
    double hi = to<double>(x.max);
    if(lo >= 0.0)
        return x;
    if(hi <= 0.0)
        return -x;
    auto neg_min = value_invoke_common([](auto v) { return -v; }, x.min);
    return {int64_t{0}, value_max(neg_min, x.max)};
}

interval floor(interval x)
{
    return {std::floor(to<double>(x.min)), std::floor(to<double>(x.max))};
}

interval ceil(interval x) { return {std::ceil(to<double>(x.min)), std::ceil(to<double>(x.max))}; }

interval pow(interval x, interval y)
{
    auto f  = MIGRAPHX_LIFT(std::pow);
    auto p1 = value_invoke_common(f, x.min, y.min);
    auto p2 = value_invoke_common(f, x.min, y.max);
    auto p3 = value_invoke_common(f, x.max, y.min);
    auto p4 = value_invoke_common(f, x.max, y.max);
    return {value_min(value_min(p1, p2), value_min(p3, p4)),
            value_max(value_max(p1, p2), value_max(p3, p4))};
}

interval min(interval x, interval y) { return {value_min(x.min, y.min), value_min(x.max, y.max)}; }

interval max(interval x, interval y) { return {value_max(x.min, y.min), value_max(x.max, y.max)}; }

expr lit(value v)
{
    return expr(literal_node{v});
}

expr var(std::string name) { return expr(variable_node{std::move(name), {}}); }

expr var(std::string name, interval constraint)
{
    return expr(variable_node{std::move(name), {std::move(constraint)}});
}

expr arg(expr x) { return x; }

expr call_op(const op_def* op, std::vector<expr> args)
{
    bool is_const = std::all_of(args.begin(), args.end(), [](const expr& e) { return e.name() == "literal"; });
    auto e = expr(op_node{op}, std::move(args));
    if(is_const)
        return lit(e.eval({}));
    return e;
}

namespace {
std::vector<expr> flatten_args(const std::string& op_name, std::vector<expr> args)
{
    std::vector<expr> flat_args;
    for(auto& a : args)
    {
        if(a.name() == op_name)
        {
            auto& c = a.children();
            flat_args.insert(flat_args.end(), c.begin(), c.end());
        }
        else
        {
            flat_args.push_back(std::move(a));
        }
    }
    return flat_args;
}
} // namespace

template <class Eval, class EvalInterval>
auto call_associative(std::string name, Eval eval, EvalInterval eval_interval)
{
    return [=](auto... es) {
        auto eval1 = [=](const std::vector<value>& args) {
            return std::accumulate(args.begin() + 1,
                                   args.end(),
                                   args.front(),
                                   [=](const value& acc, const value& arg) {
                                       return value_invoke_common(eval, acc, arg);
                                   });
        };
        auto eval_interval1 = [=](const std::vector<interval>& args) {
            return std::accumulate(
                args.begin() + 1,
                args.end(),
                args.front(),
                [=](const interval& acc, const interval& arg) { return eval_interval(acc, arg); });
        };
        return call_op(name, eval1, eval_interval1, flatten_args(name, {arg(es)...}));
    };
}

template <class Eval>
auto call_associative(std::string name, Eval eval)
{
    return call_associative(name, eval, eval);
}

expr operator+(expr ex, expr ey)
{
    return call_associative("+", [](auto x, auto y) { return x + y; })(std::move(ex),
                                                                       std::move(ey));
}

expr operator-(expr ex, expr ey)
{
    return call("-", [](auto x, auto y) { return x - y; })(std::move(ex), std::move(ey));
}

expr operator*(expr ex, expr ey)
{
    return call_associative("*", [](auto x, auto y) { return x * y; })(std::move(ex),
                                                                       std::move(ey));
}

expr operator/(expr ex, expr ey)
{
    return call("/", [](auto x, auto y) { return x / y; })(std::move(ex), std::move(ey));
}

expr operator-(expr e)
{
    return call("neg", [](auto x) { return -x; })(std::move(e));
}

bool operator==(const expr& a, const expr& b)
{
    if(a.pimpl == b.pimpl)
        return true;
    if(not a.pimpl or not b.pimpl)
        return false;
    return a.pimpl->node == b.pimpl->node and a.pimpl->children == b.pimpl->children;
}

bool operator!=(const expr& a, const expr& b) { return not(a == b); }

expr sin(expr e)
{
    return call("sin", MIGRAPHX_LIFT(std::sin), [](interval x) { return sin(x); })(std::move(e));
}

expr cos(expr e)
{
    return call("cos", MIGRAPHX_LIFT(std::cos), [](interval x) { return cos(x); })(std::move(e));
}

expr tan(expr e)
{
    return call("tan", MIGRAPHX_LIFT(std::tan), [](interval x) { return tan(x); })(std::move(e));
}

expr exp(expr e)
{
    return call("exp", MIGRAPHX_LIFT(std::exp), [](interval x) { return exp(x); })(std::move(e));
}

expr log(expr e)
{
    return call("log", MIGRAPHX_LIFT(std::log), [](interval x) { return log(x); })(std::move(e));
}

expr sqrt(expr e)
{
    return call("sqrt", MIGRAPHX_LIFT(std::sqrt), [](interval x) { return sqrt(x); })(std::move(e));
}

expr abs(expr e)
{
    return call(
        "abs", [](auto x) { return x < 0 ? -x : x; }, [](interval x) { return abs(x); })(
        std::move(e));
}

expr floor(expr e)
{
    return call("floor", MIGRAPHX_LIFT(std::floor), [](interval x) { return floor(x); })(
        std::move(e));
}

expr ceil(expr e)
{
    return call("ceil", MIGRAPHX_LIFT(std::ceil), [](interval x) { return ceil(x); })(std::move(e));
}

expr pow(expr x, expr y)
{
    return call("pow", MIGRAPHX_LIFT(std::pow), [](interval a, interval b) { return pow(a, b); })(
        std::move(x), std::move(y));
}

expr min(expr x, expr y)
{
    return call(
        "min",
        [](auto a, auto b) { return a < b ? a : b; },
        [](interval a, interval b) { return min(a, b); })(std::move(x), std::move(y));
}

expr max(expr x, expr y)
{
    return call(
        "max",
        [](auto a, auto b) { return a > b ? a : b; },
        [](interval a, interval b) { return max(a, b); })(std::move(x), std::move(y));
}

std::string expr::name() const
{
    if(auto* n = std::get_if<literal_node>(&pimpl->node))
        return "literal";
    if(auto* n = std::get_if<variable_node>(&pimpl->node))
        return "variable";
    auto* n = std::get_if<op_node>(&pimpl->node);
    return n->op->name;
}

const node_variant& expr::node() const { return pimpl->node; }

const std::vector<expr>& expr::children() const { return pimpl->children; }

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

namespace {
std::string value_to_string(const value& v)
{
    return std::visit(
        [](auto x) -> std::string {
            std::ostringstream ss;
            ss << x;
            return ss.str();
        },
        v);
}

bool is_infix_op(const std::string& name)
{
    return name == "+" or name == "-" or name == "*" or name == "/";
}
} // namespace

std::string expr::to_string() const
{
    if(auto* n = std::get_if<literal_node>(&pimpl->node))
        return value_to_string(n->val);
    if(auto* n = std::get_if<variable_node>(&pimpl->node))
        return n->name;
    auto* n = std::get_if<op_node>(&pimpl->node);
    if(n->op->name == "neg" and pimpl->children.size() == 1)
        return "(-" + pimpl->children[0].to_string() + ")";
    if(is_infix_op(n->op->name) and pimpl->children.size() >= 2)
    {
        std::string result = "(";
        for(std::size_t i = 0; i < pimpl->children.size(); ++i)
        {
            if(i > 0)
                result += " " + n->op->name + " ";
            result += pimpl->children[i].to_string();
        }
        result += ")";
        return result;
    }
    std::string result = n->op->name + "(";
    for(std::size_t i = 0; i < pimpl->children.size(); ++i)
    {
        if(i > 0)
            result += ", ";
        result += pimpl->children[i].to_string();
    }
    result += ")";
    return result;
}

std::string to_string(const expr& e) { return e.to_string(); }

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
