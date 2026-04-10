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
#include <migraphx/simple_parser.hpp>
#include <algorithm>
#include <iterator>
#include <functional>
#include <numeric>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace sym {

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

interval operator%(interval a, interval b)
{
    auto f       = [](auto x, auto y) { return x % y; };
    auto fd      = [](auto x, auto y) { return std::fmod(x, y); };
    auto compute = [&](const value& x, const value& y) -> value {
        if(std::holds_alternative<int64_t>(x) and std::holds_alternative<int64_t>(y))
            return f(std::get<int64_t>(x), std::get<int64_t>(y));
        return fd(to<double>(x), to<double>(y));
    };
    auto p1 = compute(a.min, b.min);
    auto p2 = compute(a.min, b.max);
    auto p3 = compute(a.max, b.min);
    auto p4 = compute(a.max, b.max);
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

std::ostream& operator<<(std::ostream& os, const interval& i)
{
    os << "[";
    std::visit([&](auto x) { os << x; }, i.min);
    os << ", ";
    std::visit([&](auto x) { os << x; }, i.max);
    os << "]";
    return os;
}

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

struct expr::impl
{
    node_variant node;
    std::vector<expr> children;
    bool raw_flag = false;
};

static std::string get_name(const node_variant& nv)
{
    if(auto* n = std::get_if<literal_node>(&nv))
        return "literal";
    if(auto* n = std::get_if<variable_node>(&nv))
        return "variable";
    auto* n = std::get_if<op_node>(&nv);
    return n->op->name;
}

std::shared_ptr<const expr::impl> expr::make_impl(node_variant node, std::vector<expr> children)
{
    bool raw =
        std::any_of(children.begin(), children.end(), [](const expr& e) { return e.is_raw(); });
    if(auto* v = std::get_if<variable_node>(&node))
        raw = raw or (not v->name.empty() and v->name[0] == '_');
    return std::make_shared<const impl>(impl{std::move(node), std::move(children), raw});
}

expr lit(value v) { return expr(literal_node{v}); }

expr var(std::string name) { return expr(variable_node{std::move(name), {}}); }

expr var(std::string name, interval constraint)
{
    return expr(variable_node{std::move(name), {std::move(constraint)}});
}

expr arg(expr x) { return x; }

static int node_type_index(const node_variant& nv)
{
    if(std::get_if<literal_node>(&nv))
        return 0;
    if(std::get_if<variable_node>(&nv))
        return 1;
    return 2;
}

static bool expr_less(const expr& a, const expr& b)
{
    int ai = node_type_index(a.node());
    int bi = node_type_index(b.node());
    if(ai != bi)
        return ai < bi;
    if(auto* la = std::get_if<literal_node>(&a.node()))
    {
        auto* lb = std::get_if<literal_node>(&b.node());
        return la->val < lb->val;
    }
    if(auto* va = std::get_if<variable_node>(&a.node()))
    {
        auto* vb = std::get_if<variable_node>(&b.node());
        return va->name < vb->name;
    }
    auto* oa = std::get_if<op_node>(&a.node());
    auto* ob = std::get_if<op_node>(&b.node());
    if(oa->op->name != ob->op->name)
        return oa->op->name < ob->op->name;
    return std::lexicographical_compare(a.children().begin(),
                                        a.children().end(),
                                        b.children().begin(),
                                        b.children().end(),
                                        expr_less);
}

static bool is_commutative(const std::string& name) { return name == "+" or name == "*"; }

static bool is_pvar(const expr& e)
{
    auto* v = std::get_if<variable_node>(&e.node());
    return v != nullptr and not v->name.empty() and v->name[0] == '_';
}

static bool
match_expr(const expr& pattern, const expr& e, std::unordered_map<std::string, expr>& bindings)
{
    if(is_pvar(pattern))
    {
        auto* v = std::get_if<variable_node>(&pattern.node());
        auto it = bindings.find(v->name);
        if(it != bindings.end())
            return it->second == e;
        bindings.emplace(v->name, e);
        return true;
    }
    if(pattern.node().index() != e.node().index())
        return false;
    if(auto* pl = std::get_if<literal_node>(&pattern.node()))
    {
        auto* el = std::get_if<literal_node>(&e.node());
        return pl->val == el->val;
    }
    if(auto* pv = std::get_if<variable_node>(&pattern.node()))
    {
        auto* ev = std::get_if<variable_node>(&e.node());
        return pv->name == ev->name and pv->constraints == ev->constraints;
    }
    auto* po = std::get_if<op_node>(&pattern.node());
    auto* eo = std::get_if<op_node>(&e.node());
    if(po->op->name != eo->op->name)
        return false;
    if(pattern.children().size() != e.children().size())
        return false;
    for(std::size_t i = 0; i < pattern.children().size(); ++i)
    {
        if(not match_expr(pattern.children()[i], e.children()[i], bindings))
            return false;
    }
    return true;
}

static expr substitute_expr(const expr& tmpl, const std::unordered_map<std::string, expr>& bindings)
{
    if(is_pvar(tmpl))
    {
        auto* v = std::get_if<variable_node>(&tmpl.node());
        return bindings.at(v->name);
    }
    if(tmpl.children().empty())
        return tmpl;
    auto* op_n = std::get_if<op_node>(&tmpl.node());
    std::vector<expr> new_children;
    new_children.reserve(tmpl.children().size());
    for(const auto& child : tmpl.children())
        new_children.push_back(substitute_expr(child, bindings));
    return call_op(op_n->op, std::move(new_children));
}

static bool is_zero(const value& v) { return v == value{int64_t{0}} or v == value{0.0}; }

static bool is_one(const value& v) { return v == value{int64_t{1}} or v == value{1.0}; }

struct term
{
    value coeff;
    std::vector<expr> bases;
};

static term extract_term(const expr& e)
{
    if(e.name() == "literal")
    {
        auto* n = std::get_if<literal_node>(&e.node());
        return {n->val, {}};
    }
    if(e.name() == "*")
    {
        value coeff{int64_t{1}};
        std::vector<expr> bases;
        for(const auto& child : e.children())
        {
            if(child.name() == "literal")
            {
                auto* n = std::get_if<literal_node>(&child.node());
                coeff   = value_invoke_common([](auto x, auto y) { return x * y; }, coeff, n->val);
            }
            else
            {
                bases.push_back(child);
            }
        }
        return {coeff, std::move(bases)};
    }
    return {value{int64_t{1}}, {e}};
}

static expr build_term(const term& t)
{
    if(t.bases.empty())
        return lit(t.coeff);
    expr base_product = t.bases[0];
    for(std::size_t i = 1; i < t.bases.size(); ++i)
        base_product = base_product * t.bases[i];
    if(is_one(t.coeff))
        return base_product;
    return lit(t.coeff) * base_product;
}

static bool bases_less(const std::vector<expr>& a, const std::vector<expr>& b)
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), expr_less);
}

static expr normalize_add(const op_def* op, std::vector<expr> args)
{
    std::vector<term> terms;
    terms.reserve(args.size());
    for(const auto& a : args)
        terms.push_back(extract_term(a));

    std::stable_sort(terms.begin(), terms.end(), [](const term& a, const term& b) {
        return bases_less(a.bases, b.bases);
    });

    std::vector<term> merged;
    for(const auto& t : terms)
    {
        if(not merged.empty() and merged.back().bases == t.bases)
        {
            merged.back().coeff = value_invoke_common(
                [](auto x, auto y) { return x + y; }, merged.back().coeff, t.coeff);
        }
        else
        {
            merged.push_back(t);
        }
    }

    merged.erase(std::remove_if(
                     merged.begin(), merged.end(), [](const term& t) { return is_zero(t.coeff); }),
                 merged.end());

    if(merged.empty())
        return lit(int64_t{0});
    if(merged.size() == 1)
        return build_term(merged[0]);

    std::vector<expr> result_children;
    result_children.reserve(merged.size());
    for(const auto& t : merged)
        result_children.push_back(build_term(t));
    std::stable_sort(result_children.begin(), result_children.end(), expr_less);
    return expr(op_node{op}, std::move(result_children));
}

static expr normalize_mul(const op_def* op, std::vector<expr> args)
{
    value coeff{int64_t{1}};
    std::vector<expr> non_literals;
    for(auto& a : args)
    {
        if(a.name() == "literal")
        {
            auto* n = std::get_if<literal_node>(&a.node());
            coeff   = value_invoke_common([](auto x, auto y) { return x * y; }, coeff, n->val);
        }
        else
        {
            non_literals.push_back(std::move(a));
        }
    }

    if(is_zero(coeff))
        return lit(coeff);

    std::vector<expr> factors;
    if(not is_one(coeff))
        factors.push_back(lit(coeff));
    factors.insert(factors.end(),
                   std::make_move_iterator(non_literals.begin()),
                   std::make_move_iterator(non_literals.end()));

    auto it =
        std::find_if(factors.begin(), factors.end(), [](const expr& e) { return e.name() == "+"; });
    if(it != factors.end())
    {
        auto plus_children = it->children();
        std::vector<expr> other_factors;
        for(auto i = factors.begin(); i != factors.end(); ++i)
        {
            if(i != it)
                other_factors.push_back(*i);
        }
        std::vector<expr> distributed;
        distributed.reserve(plus_children.size());
        for(const auto& pc : plus_children)
        {
            expr product = pc;
            for(const auto& f : other_factors)
                product = product * f;
            distributed.push_back(std::move(product));
        }
        expr result = distributed[0];
        for(std::size_t i = 1; i < distributed.size(); ++i)
            result = result + distributed[i];
        return result;
    }

    if(factors.empty())
        return lit(coeff);
    if(factors.size() == 1)
        return factors[0];
    std::stable_sort(factors.begin(), factors.end(), expr_less);
    return expr(op_node{op}, std::move(factors));
}

static expr normalize_impl(const op_def* op, std::vector<expr> args)
{
    bool is_const =
        std::all_of(args.begin(), args.end(), [](const expr& e) { return e.name() == "literal"; });
    if(is_const)
    {
        auto e = expr(op_node{op}, std::move(args));
        return lit(e.eval({}));
    }
    if(op->name == "+")
        return normalize_add(op, std::move(args));
    if(op->name == "*")
        return normalize_mul(op, std::move(args));
    return expr(op_node{op}, std::move(args));
}

static const std::vector<rewrite_rule>& get_rewrite_rules()
{
    static const std::vector<rewrite_rule> rules = [] {
        auto _1 = pvar(1);
        auto _2 = pvar(2);
        return std::vector<rewrite_rule>{
            sqrt(_1 * _2) >> sqrt(_1) * sqrt(_2),
            sqrt(_1 / _2) >> sqrt(_1) / sqrt(_2),
            log(exp(_1)) >> _1,
            exp(log(_1)) >> _1,
        };
    }();
    return rules;
}

static expr apply_rewrite_rules(const expr& e)
{
    for(const auto& rule : get_rewrite_rules())
    {
        std::unordered_map<std::string, expr> bindings;
        if(match_expr(rule.pattern, e, bindings))
            return substitute_expr(rule.replacement, bindings);
    }
    return e;
}

static expr normalize_expr(const op_def* op, std::vector<expr> args)
{
    return apply_rewrite_rules(normalize_impl(op, std::move(args)));
}

expr call_op(const op_def* op, std::vector<expr> args)
{
    if(std::any_of(args.begin(), args.end(), [](const expr& e) { return e.is_raw(); }))
        return expr(op_node{op}, std::move(args));
    return normalize_expr(op, std::move(args));
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

expr operator-(expr ex, expr ey) { return ex + (-ey); }

expr operator*(expr ex, expr ey)
{
    return call_associative("*", [](auto x, auto y) { return x * y; })(std::move(ex),
                                                                       std::move(ey));
}

expr operator/(expr ex, expr ey)
{
    return call("/", [](auto x, auto y) { return x / y; })(std::move(ex), std::move(ey));
}

expr operator%(expr ex, expr ey)
{
    return call(
        "%",
        [](auto x, auto y) {
            if constexpr(std::is_integral_v<decltype(x)> and std::is_integral_v<decltype(y)>)
                return x % y;
            else
                return std::fmod(static_cast<double>(x), static_cast<double>(y));
        },
        [](interval x, interval y) { return x % y; })(std::move(ex), std::move(ey));
}

expr operator-(expr e) { return lit(-1) * std::move(e); }

bool operator==(const expr& a, const expr& b)
{
    if(a.pimpl == b.pimpl)
        return true;
    if(not a.pimpl or not b.pimpl)
        return false;
    return a.pimpl->node == b.pimpl->node and a.pimpl->children == b.pimpl->children;
}

bool operator!=(const expr& a, const expr& b) { return not(a == b); }

std::ostream& operator<<(std::ostream& os, const expr& e) { return os << e.to_string(); }

bool expr::empty() const { return not pimpl; }

static std::size_t hash_combine(std::size_t seed, std::size_t h)
{
    return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

static std::size_t hash_value(const value& v)
{
    return std::visit([](auto x) -> std::size_t { return std::hash<decltype(x)>{}(x); }, v);
}

static std::size_t hash_node(const node_variant& nv)
{
    return std::visit(
        [](const auto& n) -> std::size_t {
            using T = std::decay_t<decltype(n)>;
            if constexpr(std::is_same<T, literal_node>{})
                return hash_value(n.val);
            else if constexpr(std::is_same<T, variable_node>{})
                return std::hash<std::string>{}(n.name);
            else
                return std::hash<const op_def*>{}(n.op);
        },
        nv);
}

std::size_t expr::hash() const
{
    if(not pimpl)
        return 0;
    auto result = hash_node(pimpl->node);
    for(const auto& child : pimpl->children)
        result = hash_combine(result, child.hash());
    return result;
}

std::size_t expr::eval_uint(const std::unordered_map<expr, std::size_t>& symbol_map) const
{
    auto it = symbol_map.find(*this);
    if(it != symbol_map.end())
        return it->second;
    auto v = eval({});
    return to<std::size_t>(v);
}

expr expr::subs(const std::unordered_map<expr, expr>& symbol_map) const
{
    auto it = symbol_map.find(*this);
    if(it != symbol_map.end())
        return it->second;
    if(not pimpl)
        return *this;
    if(std::holds_alternative<literal_node>(pimpl->node) or
       std::holds_alternative<variable_node>(pimpl->node))
        return *this;
    std::vector<expr> new_children;
    new_children.reserve(pimpl->children.size());
    std::transform(pimpl->children.begin(),
                   pimpl->children.end(),
                   std::back_inserter(new_children),
                   [&](const expr& child) { return child.subs(symbol_map); });
    auto* n = std::get_if<op_node>(&pimpl->node);
    return call_op(n->op, std::move(new_children));
}

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

std::string expr::name() const { return get_name(pimpl->node); }

bool expr::is_raw() const { return pimpl and pimpl->raw_flag; }

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
    return name == "+" or name == "-" or name == "*" or name == "/" or name == "%";
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

expr pvar(int id) { return var("_" + std::to_string(id)); }

namespace {

expr simplify_impl(const expr& e, const std::vector<rewrite_rule>& rules);

expr apply_rules(const expr& e, const std::vector<rewrite_rule>& rules)
{
    for(const auto& rule : rules)
    {
        std::unordered_map<std::string, expr> bindings;
        if(match_expr(rule.pattern, e, bindings))
            return simplify_impl(substitute_expr(rule.replacement, bindings), rules);
    }
    return e;
}

expr simplify_impl(const expr& e, const std::vector<rewrite_rule>& rules)
{
    if(e.children().empty())
        return apply_rules(e, rules);
    auto* op_n = std::get_if<op_node>(&e.node());
    std::vector<expr> new_children;
    new_children.reserve(e.children().size());
    for(const auto& child : e.children())
        new_children.push_back(simplify_impl(child, rules));
    return apply_rules(call_op(op_n->op, std::move(new_children)), rules);
}

} // namespace

expr simplify(expr e, std::vector<rewrite_rule> rules) { return simplify_impl(e, rules); }

namespace {

using sym_parser = parser::simple_string_view_skip_parser;

expr parse_expr(sym_parser& p);

template <class F>
struct call_wrapper
{
    F f;
    template <class... Args>
    auto try_call(rank<1>, Args&&... args) const -> decltype(f(std::forward<Args>(args)...))
    {
        return f(std::forward<Args>(args)...);
    }

    template <class... Args>
    expr try_call(rank<0>, Args&&... args) const
    {
        MIGRAPHX_THROW(
            (std::string("Function is not callable: ") + ... + (to_string(args) + ", ")));
    }

    template <class G>
    static expr visit_size(std::size_t n, G g)
    {
        switch(n)
        {
        case 0: return g(std::integral_constant<std::size_t, 0>{});
        case 1: return g(std::integral_constant<std::size_t, 1>{});
        case 2: return g(std::integral_constant<std::size_t, 2>{});
        case 3: return g(std::integral_constant<std::size_t, 3>{});
        default: MIGRAPHX_THROW("Invalid size: " + std::to_string(n));
        }
    }

    expr operator()(const std::vector<expr>& args) const
    {
        return visit_size(args.size(), [&](auto n) {
            return sequence_c<n>([&](auto... is) { return try_call(rank<1>{}, args[is]...); });
        });
    }
};

template <class F>
call_wrapper(F) -> call_wrapper<F>;

expr call_function(const std::string& name, std::vector<expr> args)
{
#define MIGRAPHX_CALL_FUNC(name)                    \
    {                                               \
        #name, call_wrapper { MIGRAPHX_LIFT(name) } \
    }
    static const std::unordered_map<std::string, std::function<expr(const std::vector<expr>& args)>>
        functions = {
            {"+", call_wrapper{std::plus<>{}}},
            {"*", call_wrapper{std::multiplies<>{}}},
            {"-", call_wrapper{std::minus<>{}}},
            {"/", call_wrapper{std::divides<>{}}},
            {"%", call_wrapper{std::modulus<>{}}},
            MIGRAPHX_CALL_FUNC(pow),
            MIGRAPHX_CALL_FUNC(min),
            MIGRAPHX_CALL_FUNC(max),
            MIGRAPHX_CALL_FUNC(sin),
            MIGRAPHX_CALL_FUNC(cos),
            MIGRAPHX_CALL_FUNC(tan),
            MIGRAPHX_CALL_FUNC(exp),
            MIGRAPHX_CALL_FUNC(log),
            MIGRAPHX_CALL_FUNC(sqrt),
            MIGRAPHX_CALL_FUNC(abs),
            MIGRAPHX_CALL_FUNC(floor),
            MIGRAPHX_CALL_FUNC(ceil),
        };
#undef MIGRAPHX_CALL_FUNC
    return functions.at(name)(args);
}

expr parse_number(sym_parser& p)
{
    if(not std::isdigit(p.peek_char()) and p.peek_char() != '.')
        return {};
    auto token    = p.parse_while([](char c) { return std::isdigit(c) or c == '.'; });
    bool is_float = token.find('.') != std::string_view::npos;
    if(is_float)
        return lit(std::stod(std::string(token)));
    return lit(std::stoll(std::string(token)));
}

expr parse_func_or_var(sym_parser& p)
{
    char c = p.peek_char();
    if(not std::isalpha(c) and c != '_')
        return {};
    auto name = p.parse_while([](char ch) { return std::isalnum(ch) or ch == '_'; });
    std::string sname(name);
    if(p.peek_char() != '(')
        return var(sname);
    p.advance(1);
    std::vector<expr> args;
    if(p.peek_char() != ')')
    {
        args.push_back(parse_expr(p));
        while(p.match(std::string_view(",")))
            args.push_back(parse_expr(p));
    }
    p.expect(std::string_view(")"));
    return call_function(sname, std::move(args));
}

expr parse_paren_expr(sym_parser& p)
{
    if(not p.match(std::string_view("(")))
        return {};
    auto e = parse_expr(p);
    p.expect(std::string_view(")"));
    return e;
}

expr parse_primary(sym_parser& p)
{
    return p.first_of(&parse_paren_expr,
                      &parse_func_or_var,
                      &parse_number,
                      [](sym_parser& q) -> expr { MIGRAPHX_THROW(q.error_message("expression")); });
}

expr parse_unary(sym_parser& p)
{
    if(p.match(std::string_view("-")))
        return -parse_unary(p);
    return parse_primary(p);
}

expr parse_mul_expr(sym_parser& p)
{
    auto left = parse_unary(p);
    auto ops  = p.repeat([](sym_parser& q) -> std::pair<std::string_view, expr> {
        auto op = q.first_of(std::string_view("*"), std::string_view("/"), std::string_view("%"));
        if(op.empty())
            return {};
        return {op, parse_unary(q)};
    });
    for(auto& [op, rhs] : ops)
        left = call_function(std::string(op), {std::move(left), std::move(rhs)});
    return left;
}

expr parse_expr(sym_parser& p)
{
    auto left = parse_mul_expr(p);
    auto ops  = p.repeat([](sym_parser& q) -> std::pair<std::string_view, expr> {
        auto op = q.first_of(std::string_view("+"), std::string_view("-"));
        if(op.empty())
            return {};
        return {op, parse_mul_expr(q)};
    });
    for(auto& [op, rhs] : ops)
        left = call_function(std::string(op), {std::move(left), std::move(rhs)});
    return left;
}

} // namespace

expr parse(const std::string& str)
{
    std::string_view sv(str);
    sym_parser p{sv};
    // skip leading whitespace
    p.advance(0);
    auto result = parse_expr(p);
    if(not p.done())
        MIGRAPHX_THROW(p.error_message("end of input"));
    return result;
}

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
