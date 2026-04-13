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
#include <migraphx/serialize.hpp>
#include <migraphx/simple_parser.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/utility_operators.hpp>
#include <migraphx/float_equal.hpp>
#include <algorithm>
#include <iterator>
#include <functional>
#include <numeric>
#include <optional>
#include <sstream>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace sym {

scalar scalar_min(const scalar& a, const scalar& b)
{
    return scalar_invoke_common([](auto x, auto y) { return x < y ? x : y; }, a, b);
}

scalar scalar_max(const scalar& a, const scalar& b)
{
    return scalar_invoke_common([](auto x, auto y) { return x > y ? x : y; }, a, b);
}

interval operator+(interval a, interval b)
{
    auto f = [](auto x, auto y) { return x + y; };
    return {scalar_invoke_common(f, a.min, b.min), scalar_invoke_common(f, a.max, b.max)};
}

interval operator-(interval a, interval b)
{
    auto f = [](auto x, auto y) { return x - y; };
    return {scalar_invoke_common(f, a.min, b.max), scalar_invoke_common(f, a.max, b.min)};
}

interval operator*(interval a, interval b)
{
    auto f  = [](auto x, auto y) { return x * y; };
    auto p1 = scalar_invoke_common(f, a.min, b.min);
    auto p2 = scalar_invoke_common(f, a.min, b.max);
    auto p3 = scalar_invoke_common(f, a.max, b.min);
    auto p4 = scalar_invoke_common(f, a.max, b.max);
    return {scalar_min(scalar_min(p1, p2), scalar_min(p3, p4)),
            scalar_max(scalar_max(p1, p2), scalar_max(p3, p4))};
}

interval operator/(interval a, interval b)
{
    auto f  = [](auto x, auto y) { return x / y; };
    auto p1 = scalar_invoke_common(f, a.min, b.min);
    auto p2 = scalar_invoke_common(f, a.min, b.max);
    auto p3 = scalar_invoke_common(f, a.max, b.min);
    auto p4 = scalar_invoke_common(f, a.max, b.max);
    return {scalar_min(scalar_min(p1, p2), scalar_min(p3, p4)),
            scalar_max(scalar_max(p1, p2), scalar_max(p3, p4))};
}

interval operator%(interval a, interval b)
{
    auto f       = [](auto x, auto y) { return x % y; };
    auto fd      = [](auto x, auto y) { return std::fmod(x, y); };
    auto compute = [&](const scalar& x, const scalar& y) -> scalar {
        if(std::holds_alternative<int64_t>(x) and std::holds_alternative<int64_t>(y))
            return f(std::get<int64_t>(x), std::get<int64_t>(y));
        return fd(to<double>(x), to<double>(y));
    };
    auto p1 = compute(a.min, b.min);
    auto p2 = compute(a.min, b.max);
    auto p3 = compute(a.max, b.min);
    auto p4 = compute(a.max, b.max);
    return {scalar_min(scalar_min(p1, p2), scalar_min(p3, p4)),
            scalar_max(scalar_max(p1, p2), scalar_max(p3, p4))};
}

interval operator-(interval a)
{
    auto f = [](auto x) { return -x; };
    return {scalar_invoke_common(f, a.max), scalar_invoke_common(f, a.min)};
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

static bool scalar_less(const scalar& a, const scalar& b)
{
    auto f = [](auto x, auto y) -> int64_t { return x < y ? 1 : 0; };
    return std::get<int64_t>(scalar_invoke_common(f, a, b)) != 0;
}

bool operator<(interval a, interval b) { return scalar_less(a.max, b.min); }

bool operator<=(interval a, interval b) { return not scalar_less(b.min, a.max); }

bool operator>(interval a, interval b) { return scalar_less(b.max, a.min); }

bool operator>=(interval a, interval b) { return not scalar_less(a.min, b.max); }

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
    auto neg_min = scalar_invoke_common([](auto v) { return -v; }, x.min);
    return {int64_t{0}, scalar_max(neg_min, x.max)};
}

interval floor(interval x)
{
    return {std::floor(to<double>(x.min)), std::floor(to<double>(x.max))};
}

interval ceil(interval x) { return {std::ceil(to<double>(x.min)), std::ceil(to<double>(x.max))}; }

interval pow(interval x, interval y)
{
    auto f  = MIGRAPHX_LIFT(std::pow);
    auto p1 = scalar_invoke_common(f, x.min, y.min);
    auto p2 = scalar_invoke_common(f, x.min, y.max);
    auto p3 = scalar_invoke_common(f, x.max, y.min);
    auto p4 = scalar_invoke_common(f, x.max, y.max);
    return {scalar_min(scalar_min(p1, p2), scalar_min(p3, p4)),
            scalar_max(scalar_max(p1, p2), scalar_max(p3, p4))};
}

interval min(interval x, interval y)
{
    return {scalar_min(x.min, y.min), scalar_min(x.max, y.max)};
}

interval max(interval x, interval y)
{
    return {scalar_max(x.min, y.min), scalar_max(x.max, y.max)};
}

struct literal_node
{
    scalar val;
    friend bool operator==(const literal_node& a, const literal_node& b)
    {
        return scalar_invoke_common<bool>(
            [](auto a, auto b) { return float_equal(a, b); }, a.val, b.val);
    }
    friend bool operator!=(const literal_node& a, const literal_node& b) { return not(a == b); }
};

struct variable_node
{
    std::string name;
    std::vector<interval> constraints;
    std::set<scalar> optimals;
    friend bool operator==(const variable_node& a, const variable_node& b)
    {
        return a.name == b.name and a.constraints == b.constraints and a.optimals == b.optimals;
    }
    friend bool operator!=(const variable_node& a, const variable_node& b) { return not(a == b); }
};

struct op_node
{
    const op_def* op;
    friend bool operator==(const op_node& a, const op_node& b) { return a.op == b.op; }
    friend bool operator!=(const op_node& a, const op_node& b) { return not(a == b); }
};

using node_variant = std::variant<literal_node, variable_node, op_node>;

static std::size_t hash_combine(std::size_t seed, std::size_t h)
{
    return seed ^ (h + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

static std::size_t hash_scalar(const scalar& v)
{
    return std::visit([](auto x) -> std::size_t { return std::hash<decltype(x)>{}(x); }, v);
}

static std::size_t hash_node(const node_variant& nv)
{
    return std::visit(
        overloaded{[](const literal_node& n) { return hash_scalar(n.val); },
                   [](const variable_node& n) { return std::hash<std::string>{}(n.name); },
                   [](const op_node& n) { return std::hash<const op_def*>{}(n.op); }},
        nv);
}

static std::size_t hash_children(const std::vector<expr>& children, std::size_t start)
{
    return transform_accumulate(
        children.begin(), children.end(), start, hash_combine, [](const expr& child) {
            return child.hash();
        });
}

struct expr::impl
{
    node_variant node;
    std::vector<expr> children;
    bool raw_flag           = false;
    std::size_t cached_hash = 0;
};

const expr::impl* expr::get_pimpl() const { return pimpl.get(); }

static const node_variant& get_node(const expr& e)
{
    assert(e.get_pimpl() != nullptr);
    return e.get_pimpl()->node;
}

static std::string get_sym_name(const node_variant& nv)
{
    return std::visit(overloaded{[](const variable_node& n) { return n.name; },
                                 [](const op_node& n) -> std::string { return n.op->name; },
                                 [](const literal_node&) -> std::string { return ""; }},
                      nv);
}

static std::string get_node_name(const node_variant& nv)
{
    return std::visit(overloaded{[](const literal_node&) -> std::string { return "literal"; },
                                 [](const variable_node&) -> std::string { return "variable"; },
                                 [](const op_node& n) -> std::string { return n.op->name; }},
                      nv);
}

static scalar get_scalar_or(const node_variant& nv, scalar s)
{
    return std::visit(
        overloaded{[](const literal_node& n) { return n.val; }, [&](const auto&) { return s; }},
        nv);
}

template <class Node>
std::shared_ptr<const expr::impl> expr::make_impl(Node node, std::vector<expr> children)
{
    bool raw =
        std::any_of(children.begin(), children.end(), [](const expr& e) { return e.is_raw(); });
    if constexpr(std::is_same<Node, variable_node>{})
        raw = raw or (not node.name.empty() and node.name[0] == '_');
    auto h = hash_children(children, hash_node(node));
    return std::make_shared<const impl>(
        impl{node_variant{std::move(node)}, std::move(children), raw, h});
}

template std::shared_ptr<const expr::impl> expr::make_impl(literal_node, std::vector<expr>);
template std::shared_ptr<const expr::impl> expr::make_impl(variable_node, std::vector<expr>);
template std::shared_ptr<const expr::impl> expr::make_impl(op_node, std::vector<expr>);

expr lit(scalar v) { return expr(literal_node{v}); }

expr var(std::string name)
{
    if(name.empty())
        MIGRAPHX_THROW("Variable name must not be empty");
    return expr(variable_node{std::move(name), {}, {}});
}

expr var(std::string name, interval constraint, std::set<scalar> optimals)
{
    if(name.empty())
        MIGRAPHX_THROW("Variable name must not be empty");
    return expr(variable_node{std::move(name), {constraint}, std::move(optimals)});
}

expr arg(expr x) { return x; }

static bool expr_children_less(const std::vector<expr>& a, const std::vector<expr>& b);

static auto expr_compare_key(const expr& e)
{
    const auto& n = get_node(e);
    auto children = make_ordered_as(std::cref(e.children()), &expr_children_less);
    return std::make_tuple(
        n.index(), get_scalar_or(n, scalar{int64_t{0}}), get_sym_name(n), children);
}

static bool expr_children_less(const std::vector<expr>& a, const std::vector<expr>& b)
{
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(), by(std::less<>{}, &expr_compare_key));
}

static bool is_pvar(const expr& e)
{
    const auto* v = std::get_if<variable_node>(&get_node(e));
    return v != nullptr and not v->name.empty() and v->name[0] == '_';
}

static bool match_expr(const expr& pattern, const expr& e, std::unordered_map<expr, expr>& bindings)
{
    if(is_pvar(pattern))
    {
        auto it = bindings.find(pattern);
        if(it != bindings.end())
            return it->second == e;
        bindings.emplace(pattern, e);
        return true;
    }
    if(get_node(pattern).index() != get_node(e).index())
        return false;
    return std::visit(overloaded{[&](const literal_node& pl) {
                                     return pl.val == std::get<literal_node>(get_node(e)).val;
                                 },
                                 [&](const variable_node& pv) {
                                     const auto& ev = std::get<variable_node>(get_node(e));
                                     return pv.name == ev.name and pv.constraints == ev.constraints;
                                 },
                                 [&](const op_node& po) {
                                     const auto& eo = std::get<op_node>(get_node(e));
                                     if(po.op->name != eo.op->name)
                                         return false;
                                     if(pattern.children().size() != e.children().size())
                                         return false;
                                     return std::equal(pattern.children().begin(),
                                                       pattern.children().end(),
                                                       e.children().begin(),
                                                       [&](const expr& p, const expr& c) {
                                                           return match_expr(p, c, bindings);
                                                       });
                                 }},
                      get_node(pattern));
}

static bool is_zero(const scalar& v) { return v == scalar{int64_t{0}} or v == scalar{0.0}; }

static bool is_one(const scalar& v) { return v == scalar{int64_t{1}} or v == scalar{1.0}; }

struct term
{
    scalar coeff;
    std::vector<expr> bases;
};

static term extract_term(const expr& e)
{
    if(e.name() == "literal")
    {
        const auto* n = std::get_if<literal_node>(&get_node(e));
        return {n->val, {}};
    }
    if(e.name() == "*")
    {
        return std::accumulate(e.children().begin(),
                               e.children().end(),
                               term{scalar{int64_t{1}}, {}},
                               [](term t, const expr& child) {
                                   if(child.name() == "literal")
                                   {
                                       const auto* n = std::get_if<literal_node>(&get_node(child));
                                       t.coeff       = scalar_invoke_common(
                                           [](auto x, auto y) { return x * y; }, t.coeff, n->val);
                                   }
                                   else
                                   {
                                       t.bases.push_back(child);
                                   }
                                   return t;
                               });
    }
    return {scalar{int64_t{1}}, {e}};
}

static expr build_term(const term& t)
{
    if(t.bases.empty())
        return lit(t.coeff);
    auto base_product = std::accumulate(t.bases.begin() + 1,
                                        t.bases.end(),
                                        t.bases.front(),
                                        [](expr acc, const expr& b) { return std::move(acc) * b; });
    if(is_one(t.coeff))
        return base_product;
    return lit(t.coeff) * base_product;
}

static expr normalize_add(const op_def* op, std::vector<expr> args)
{
    std::vector<term> terms;
    terms.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(terms), extract_term);

    std::stable_sort(terms.begin(), terms.end(), [](const term& a, const term& b) {
        return expr_children_less(a.bases, b.bases);
    });

    // Merge adjacent terms with matching bases
    std::vector<term> merged;
    group_unique(
        terms.begin(),
        terms.end(),
        [&](auto first, auto last) {
            merged.push_back(
                std::accumulate(std::next(first), last, *first, [](term acc, const term& t) {
                    acc.coeff = scalar_invoke_common(
                        [](auto x, auto y) { return x + y; }, acc.coeff, t.coeff);
                    return acc;
                }));
        },
        [](const term& a, const term& b) { return a.bases == b.bases; });

    merged.erase(std::remove_if(
                     merged.begin(), merged.end(), [](const term& t) { return is_zero(t.coeff); }),
                 merged.end());

    if(merged.empty())
        return lit(int64_t{0});
    if(merged.size() == 1)
        return build_term(merged[0]);

    std::vector<expr> result_children;
    result_children.reserve(merged.size());
    std::transform(merged.begin(), merged.end(), std::back_inserter(result_children), build_term);
    std::stable_sort(
        result_children.begin(), result_children.end(), by(std::greater<>{}, &expr_compare_key));
    return expr(op_node{op}, std::move(result_children));
}

static expr normalize_mul(const op_def* op, std::vector<expr> args)
{
    auto partition_it = std::stable_partition(
        args.begin(), args.end(), [](const expr& a) { return a.name() != "literal"; });
    auto coeff = transform_accumulate(
        partition_it,
        args.end(),
        scalar{int64_t{1}},
        [](scalar acc, scalar v) {
            return scalar_invoke_common([](auto x, auto y) { return x * y; }, acc, v);
        },
        [](const expr& a) { return std::get_if<literal_node>(&get_node(a))->val; });

    if(is_zero(coeff))
        return lit(coeff);

    std::vector<expr> factors;
    if(not is_one(coeff))
        factors.push_back(lit(coeff));
    factors.insert(factors.end(),
                   std::make_move_iterator(args.begin()),
                   std::make_move_iterator(partition_it));

    auto it =
        std::find_if(factors.begin(), factors.end(), [](const expr& e) { return e.name() == "+"; });
    if(it != factors.end())
    {
        auto plus_children = it->children();
        std::vector<expr> other_factors;
        std::copy_if(factors.begin(),
                     factors.end(),
                     std::back_inserter(other_factors),
                     [&](const expr& f) { return &f != &*it; });
        std::vector<expr> distributed;
        distributed.reserve(plus_children.size());
        std::transform(plus_children.begin(),
                       plus_children.end(),
                       std::back_inserter(distributed),
                       [&](const expr& pc) {
                           return std::accumulate(
                               other_factors.begin(),
                               other_factors.end(),
                               pc,
                               [](expr product, const expr& f) { return std::move(product) * f; });
                       });
        return std::accumulate(distributed.begin() + 1,
                               distributed.end(),
                               distributed.front(),
                               [](expr acc, const expr& e) { return std::move(acc) + e; });
    }

    if(factors.empty())
        return lit(coeff);
    if(factors.size() == 1)
        return factors[0];
    std::stable_sort(factors.begin(), factors.end(), by(std::less<>{}, &expr_compare_key));
    return expr(op_node{op}, std::move(factors));
}

static expr normalize_div(const op_def* op, std::vector<expr> args)
{
    const auto& num = args[0];
    const auto& den = args[1];

    // 0 / x == 0
    if(num.name() == "literal")
    {
        const auto* n = std::get_if<literal_node>(&get_node(num));
        if(is_zero(n->val))
            return lit(n->val);
    }

    // x / 1 == x
    if(den.name() == "literal")
    {
        const auto* n = std::get_if<literal_node>(&get_node(den));
        if(is_one(n->val))
            return num;
    }

    // x / x == 1
    if(num == den)
        return lit(int64_t{1});

    // Factor cancellation between products
    auto num_term = extract_term(num);
    auto den_term = extract_term(den);

    // Cancel common symbolic bases using set_difference on sorted ranges
    auto num_bases = num_term.bases;
    auto den_bases = den_term.bases;
    auto cmp       = by(std::less<>{}, &expr_compare_key);
    std::stable_sort(num_bases.begin(), num_bases.end(), cmp);
    std::stable_sort(den_bases.begin(), den_bases.end(), cmp);

    std::vector<expr> remaining_num_bases;
    std::set_difference(num_bases.begin(),
                        num_bases.end(),
                        den_bases.begin(),
                        den_bases.end(),
                        std::back_inserter(remaining_num_bases),
                        cmp);
    std::vector<expr> remaining_den_bases;
    std::set_difference(den_bases.begin(),
                        den_bases.end(),
                        num_bases.begin(),
                        num_bases.end(),
                        std::back_inserter(remaining_den_bases),
                        cmp);

    bool bases_changed = remaining_num_bases.size() != num_term.bases.size() or
                         remaining_den_bases.size() != den_term.bases.size();

    // Cancel GCD of integer coefficients
    auto num_coeff       = num_term.coeff;
    auto den_coeff       = den_term.coeff;
    scalar new_num_coeff = num_coeff;
    scalar new_den_coeff = den_coeff;

    if(std::holds_alternative<int64_t>(num_coeff) and std::holds_alternative<int64_t>(den_coeff))
    {
        auto nc = std::get<int64_t>(num_coeff);
        auto dc = std::get<int64_t>(den_coeff);
        if(dc != 0)
        {
            auto g = std::gcd(std::abs(nc), std::abs(dc));
            if(g > 1)
            {
                new_num_coeff = int64_t{nc / g};
                new_den_coeff = int64_t{dc / g};
                bases_changed = true;
            }
        }
    }

    if(bases_changed)
    {
        expr new_num = build_term({new_num_coeff, remaining_num_bases});
        expr new_den = build_term({new_den_coeff, remaining_den_bases});

        if(new_den.name() == "literal")
        {
            const auto* n = std::get_if<literal_node>(&get_node(new_den));
            if(is_one(n->val))
                return new_num;
        }
        return new_num / new_den;
    }

    // Distribute over sum: (a*k + b*k) / k when all terms are divisible
    if(num.name() == "+" and den.name() == "literal")
    {
        const auto* d = std::get_if<literal_node>(&get_node(den));
        if(std::holds_alternative<int64_t>(d->val))
        {
            auto dv = std::get<int64_t>(d->val);
            bool all_divisible =
                std::all_of(num.children().begin(), num.children().end(), [&](const expr& child) {
                    auto t = extract_term(child);
                    if(not std::holds_alternative<int64_t>(t.coeff))
                        return false;
                    return std::get<int64_t>(t.coeff) % dv == 0;
                });
            if(all_divisible)
            {
                std::vector<expr> divided;
                divided.reserve(num.children().size());
                std::transform(num.children().begin(),
                               num.children().end(),
                               std::back_inserter(divided),
                               [&](const expr& child) { return child / den; });
                return std::accumulate(divided.begin() + 1,
                                       divided.end(),
                                       divided.front(),
                                       [](expr acc, const expr& e) { return std::move(acc) + e; });
            }
        }
    }

    return expr(op_node{op}, std::move(args));
}

static expr normalize_impl(const op_def* op, std::vector<expr> args)
{
    if(std::any_of(args.begin(), args.end(), [](const expr& e) { return e.empty(); }))
    {
        return {};
    }
    if(std::all_of(args.begin(), args.end(), [](const expr& e) { return e.name() == "literal"; }))
    {
        auto e = expr(op_node{op}, std::move(args));
        return lit(e.eval({}));
    }
    if(contains({"/", "%"}, op->name) and args.at(1) == lit(0))
        MIGRAPHX_THROW("Division by zero");
    if(op->name == "+")
        return normalize_add(op, std::move(args));
    if(op->name == "*")
        return normalize_mul(op, std::move(args));
    if(op->name == "/")
        return normalize_div(op, std::move(args));
    return expr(op_node{op}, std::move(args));
}

static const std::vector<rewrite_rule>& get_rewrite_rules()
{
    static const std::vector<rewrite_rule> rules = [] {
        auto _1 = pvar(1); // NOLINT(readability-identifier-naming)
        auto _2 = pvar(2); // NOLINT(readability-identifier-naming)
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
    if(e.empty())
        return e;
    for(const auto& rule : get_rewrite_rules())
    {
        std::unordered_map<expr, expr> bindings;
        if(match_expr(rule.pattern, e, bindings))
            return rule.replacement.subs(bindings);
    }
    return e;
}

static expr normalize_expr(const op_def* op, std::vector<expr> args)
{
    return apply_rewrite_rules(normalize_impl(op, std::move(args)));
}

static std::vector<expr> flatten_args(const std::string& op_name, std::vector<expr> args)
{
    std::vector<expr> flat_args;
    std::transform(args.begin(), args.end(), join_back_inserter(flat_args), [&](const expr& a) {
        if(a.name() == op_name)
            return a.children();
        return std::vector<expr>{a};
    });
    return flat_args;
}

static expr fold_associative_args(expr e)
{
    if(e.empty())
        return e;
    if(not std::holds_alternative<op_node>(get_node(e)))
        return e;
    if(e.children().size() <= 2)
        return e;
    const auto& op_n = std::get<op_node>(get_node(e));
    auto children    = std::accumulate(e.children().begin() + 1,
                                    e.children().end(),
                                    std::vector<expr>{e.children().front()},
                                    [&](std::vector<expr> c, expr x) {
                                        if(std::holds_alternative<literal_node>(get_node(x)) and
                                           std::holds_alternative<literal_node>(get_node(c.back())))
                                        {
                                            auto d   = expr(op_n, {c.back(), x});
                                            c.back() = lit(d.eval({}));
                                        }
                                        else
                                        {
                                            c.push_back(std::move(x));
                                        }
                                        return c;
                                    });
    return expr(op_n, std::move(children));
}

expr call_op(const op_def* op, std::vector<expr> args)
{
    if(std::any_of(args.begin(), args.end(), [](const expr& e) { return e.is_raw(); }))
        return expr(op_node{op}, std::move(args));
    if(op->associative)
        args = flatten_args(op->name, std::move(args));
    auto result = normalize_expr(op, std::move(args));
    if(op->associative)
        result = fold_associative_args(std::move(result));
    return result;
}

template <class Eval, class EvalInterval>
static auto call_associative(std::string name, Eval eval, EvalInterval eval_interval)
{
    return [=](auto... es) {
        auto eval1 = [=](const std::vector<scalar>& args) {
            return std::accumulate(args.begin() + 1,
                                   args.end(),
                                   args.front(),
                                   [=](const scalar& acc, const scalar& arg) {
                                       return scalar_invoke_common(eval, acc, arg);
                                   });
        };
        auto eval_interval1 = [=](const std::vector<interval>& args) {
            return std::accumulate(
                args.begin() + 1,
                args.end(),
                args.front(),
                [=](const interval& acc, const interval& arg) { return eval_interval(acc, arg); });
        };
        return call_op(name, eval1, eval_interval1, {arg(std::move(es))...}, true);
    };
}

template <class Eval>
static auto call_associative(std::string name, Eval eval)
{
    return call_associative(std::move(name), eval, eval);
}

expr operator+(expr ex, expr ey)
{
    return call_associative("+", [](auto x, auto y) { return x + y; })(std::move(ex),
                                                                       std::move(ey));
}

expr operator-(expr ex, expr ey) { return std::move(ex) + (-std::move(ey)); }

expr operator*(expr ex, expr ey)
{
    return call_associative("*", [](auto x, auto y) { return x * y; })(std::move(ex),
                                                                       std::move(ey));
}

expr operator/(expr ex, expr ey)
{
    return call(
        "/",
        [](auto x, auto y) {
            if(float_equal(y, 0))
                MIGRAPHX_THROW("Division by zero");
            return x / y;
        },
        [](interval x, interval y) { return x / y; })(std::move(ex), std::move(ey));
}

expr operator%(expr ex, expr ey)
{
    return call(
        "%",
        [](auto x, auto y) {
            if(float_equal(y, 0))
                MIGRAPHX_THROW("Division by zero");
            if constexpr(std::is_integral<decltype(x)>{} and std::is_integral<decltype(y)>{})
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
    if(a.pimpl->cached_hash != b.pimpl->cached_hash)
        return false;
    return get_node(a) == get_node(b) and a.children() == b.children();
}

bool operator!=(const expr& a, const expr& b) { return not(a == b); }

std::ostream& operator<<(std::ostream& os, const expr& e) { return os << e.to_string(); }

bool expr::empty() const { return not pimpl; }

std::size_t expr::hash() const
{
    if(not pimpl)
        return 0;
    return pimpl->cached_hash;
}

static scalar generic_eval_auto_apply(const op_node& op, const std::vector<scalar>& args)
{
    return op.op->eval(args);
}

static interval generic_eval_auto_apply(const op_node& op, const std::vector<interval>& args)
{
    return op.op->eval_interval(args);
}

static expr generic_eval_auto_apply(const op_node& op, const std::vector<expr>& args)
{
    return call_op(op.op, args);
}

template <class R, class Replace, class Apply>
static R generic_eval(const expr& e, const Replace& replace, const Apply& apply)
{
    auto r = replace(e);
    if(r)
        return *r;
    const auto& children = e.children();
    std::vector<R> args;
    args.reserve(children.size());
    std::transform(children.begin(),
                   children.end(),
                   std::back_inserter(args),
                   [&](const expr& child) { return generic_eval<R>(child, replace, apply); });
    return apply(std::get<op_node>(get_node(e)), std::move(args));
}

template <class R, class Replace>
static R generic_eval(const expr& e, const Replace& replace)
{
    return generic_eval<R>(e, replace, MIGRAPHX_LIFT(generic_eval_auto_apply));
}

std::size_t expr::eval_uint(const std::unordered_map<expr, std::size_t>& symbol_map) const
{
    return to<std::size_t>(generic_eval<scalar>(*this, [&](const expr& e) -> std::optional<scalar> {
        auto it = symbol_map.find(e);
        if(it != symbol_map.end())
            return make_scalar(it->second);
        return std::visit(
            overloaded{[](const literal_node& n) -> std::optional<scalar> { return n.val; },
                       [](const auto&) -> std::optional<scalar> { return std::nullopt; }},
            get_node(e));
    }));
}

expr expr::subs(const std::unordered_map<expr, expr>& symbol_map) const
{
    return generic_eval<expr>(*this, [&](const expr& e) -> std::optional<expr> {
        auto it = symbol_map.find(e);
        if(it != symbol_map.end())
            return it->second;
        if(e.empty())
            return e;
        return std::visit(
            overloaded{[&](const literal_node&) -> std::optional<expr> { return e; },
                       [&](const variable_node&) -> std::optional<expr> { return e; },
                       [](const op_node&) -> std::optional<expr> { return std::nullopt; }},
            get_node(e));
    });
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

std::string expr::name() const
{
    if(empty())
        return "";
    return get_node_name(get_node(*this));
}

bool expr::is_raw() const { return pimpl and pimpl->raw_flag; }

const std::vector<expr>& expr::children() const { return pimpl->children; }

scalar expr::eval(const std::unordered_map<expr, scalar>& vars) const
{
    return generic_eval<scalar>(*this, [&](const expr& e) -> std::optional<scalar> {
        auto it = vars.find(e);
        if(it != vars.end())
            return it->second;
        return std::visit(
            overloaded{[](const literal_node& n) -> std::optional<scalar> { return n.val; },
                       [](const auto&) -> std::optional<scalar> { return std::nullopt; }},
            get_node(e));
    });
}

interval expr::eval_interval(const std::unordered_map<expr, interval>& vars) const
{
    return generic_eval<interval>(*this, [&](const expr& e) -> std::optional<interval> {
        auto it = vars.find(e);
        if(it != vars.end())
            return it->second;
        return std::visit(
            overloaded{[](const literal_node& n) -> std::optional<interval> {
                           return interval{n.val, n.val};
                       },
                       [](const variable_node& n) -> std::optional<interval> {
                           if(not n.constraints.empty())
                               return n.constraints.front();
                           MIGRAPHX_THROW("Variable '" + n.name + "' not found in interval map");
                       },
                       [](const op_node&) -> std::optional<interval> { return std::nullopt; }},
            get_node(e));
    });
}

static void collect_optimals(const expr& e, std::unordered_map<expr, std::set<scalar>>& result)
{
    if(e.empty())
        return;
    std::visit(overloaded{[&](const variable_node& n) {
                              if(not n.optimals.empty())
                                  result[e].insert(n.optimals.begin(), n.optimals.end());
                          },
                          [](const auto&) {}},
               get_node(e));
    for(const auto& child : e.children())
        collect_optimals(child, result);
}

std::set<scalar> expr::eval_optimals() const
{
    std::unordered_map<expr, std::set<scalar>> var_optimals;
    collect_optimals(*this, var_optimals);

    if(var_optimals.empty())
        return {eval({})};

    std::vector<std::pair<expr, std::vector<scalar>>> var_values;
    var_values.reserve(var_optimals.size());
    std::transform(var_optimals.begin(),
                   var_optimals.end(),
                   std::back_inserter(var_values),
                   [](const auto& p) {
                       return std::make_pair(p.first,
                                             std::vector<scalar>(p.second.begin(), p.second.end()));
                   });

    std::set<scalar> results;
    std::unordered_map<expr, scalar> current;
    fix<void>([&](auto self, std::size_t index) {
        if(index == var_values.size())
        {
            results.insert(eval(current));
            return;
        }
        const auto& [var_expr, values] = var_values[index];
        for(const auto& v : values)
        {
            current[var_expr] = v;
            self(index + 1);
        }
    })(0);
    return results;
}

static std::string scalar_to_string(const scalar& v)
{
    return std::visit(
        [](auto x) -> std::string {
            std::ostringstream ss;
            ss << x;
            return ss.str();
        },
        v);
}

struct string_prec
{
    std::string str;
    int prec = 0;
};

static int op_precedence(const std::string& name)
{
    if(name == "+" or name == "-")
        return 1;
    if(name == "*" or name == "/" or name == "%")
        return 2;
    return 0;
}

static bool is_infix_op(const std::string& name) { return op_precedence(name) > 0; }

static std::string wrap_if(const string_prec& sp, int parent_prec)
{
    if(sp.prec > 0 and sp.prec < parent_prec)
        return "(" + sp.str + ")";
    return sp.str;
}

std::string expr::to_string() const
{
    return generic_eval<string_prec>(
               *this,
               [](const expr& e) -> std::optional<string_prec> {
                   if(e.empty())
                       return string_prec{};
                   return std::visit(
                       overloaded{[](const literal_node& n) -> std::optional<string_prec> {
                                      return string_prec{scalar_to_string(n.val)};
                                  },
                                  [](const variable_node& n) -> std::optional<string_prec> {
                                      return string_prec{n.name};
                                  },
                                  [](const op_node&) -> std::optional<string_prec> {
                                      return std::nullopt;
                                  }},
                       get_node(e));
               },
               [](const op_node& op, std::vector<string_prec> args) -> string_prec {
                   int prec = op_precedence(op.op->name);
                   if(is_infix_op(op.op->name) and args.size() >= 2)
                   {
                       // -1*x -> -x
                       if(op.op->name == "*" and args[0].str == "-1")
                       {
                           std::vector<std::string> strs;
                           strs.reserve(args.size() - 1);
                           std::transform(args.begin() + 1,
                                          args.end(),
                                          std::back_inserter(strs),
                                          [&](const string_prec& sp) { return wrap_if(sp, prec); });
                           return {"-" + join_strings(strs, "*"), prec};
                       }
                       // x + (-y) -> x - y
                       if(op.op->name == "+")
                       {
                           std::string result = wrap_if(args[0], prec);
                           std::for_each(args.begin() + 1, args.end(), [&](const string_prec& sp) {
                               auto s = wrap_if(sp, prec);
                               if(not s.empty() and s.front() == '-')
                                   result += " - " + s.substr(1);
                               else
                                   result += " + " + s;
                           });
                           return {result, prec};
                       }
                       std::string delim = prec >= 2 ? op.op->name : " " + op.op->name + " ";
                       std::vector<std::string> strs;
                       strs.reserve(args.size());
                       std::transform(args.begin(),
                                      args.end(),
                                      std::back_inserter(strs),
                                      [&](const string_prec& sp) { return wrap_if(sp, prec); });
                       return {join_strings(strs, delim), prec};
                   }
                   std::vector<std::string> strs;
                   strs.reserve(args.size());
                   std::transform(args.begin(),
                                  args.end(),
                                  std::back_inserter(strs),
                                  [](const string_prec& sp) { return sp.str; });
                   return {op.op->name + "(" + join_strings(strs, ", ") + ")"};
               })
        .str;
}

std::string to_string(const expr& e) { return e.to_string(); }

expr pvar(int id) { return var("_" + std::to_string(id)); }

static expr simplify_impl(const expr& e, const std::vector<rewrite_rule>& rules);

static expr apply_rules(const expr& e, const std::vector<rewrite_rule>& rules)
{
    for(const auto& rule : rules)
    {
        std::unordered_map<expr, expr> bindings;
        if(match_expr(rule.pattern, e, bindings))
            return simplify_impl(rule.replacement.subs(bindings), rules);
    }
    return e;
}

static expr simplify_impl(const expr& e, const std::vector<rewrite_rule>& rules)
{
    if(e.children().empty())
        return apply_rules(e, rules);
    const auto* op_n = std::get_if<op_node>(&get_node(e));
    std::vector<expr> new_children;
    new_children.reserve(e.children().size());
    std::transform(e.children().begin(),
                   e.children().end(),
                   std::back_inserter(new_children),
                   [&](const expr& child) { return simplify_impl(child, rules); });
    return apply_rules(call_op(op_n->op, std::move(new_children)), rules);
}

expr simplify(const expr& e, const std::vector<rewrite_rule>& rules)
{
    return simplify_impl(e, rules);
}

using sym_parser = parser::simple_string_view_skip_parser;

static expr parse_expr(sym_parser& p);

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

template <class F>
static auto associative_call_wrapper(F f)
{
    return [=](const std::vector<expr>& args) {
        if(args.empty())
            MIGRAPHX_THROW("Associative function requires at least one argument");
        return std::accumulate(args.begin() + 1, args.end(), args.front(), f);
    };
}

static expr call_function(const std::string& name, const std::vector<expr>& args)
{
#define MIGRAPHX_CALL_FUNC(name)                    \
    {                                               \
        #name, call_wrapper { MIGRAPHX_LIFT(name) } \
    }
    static const std::unordered_map<std::string, std::function<expr(const std::vector<expr>& args)>>
        functions = {
            {"+", associative_call_wrapper(std::plus<>{})},
            {"*", associative_call_wrapper(std::multiplies<>{})},
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

static expr parse_number(sym_parser& p)
{
    if((std::isdigit(p.peek_char()) == 0) and p.peek_char() != '.')
        return {};
    auto token    = p.parse_while([](char c) { return std::isdigit(c) or c == '.'; });
    bool is_float = token.find('.') != std::string_view::npos;
    if(is_float)
        return lit(std::stod(std::string(token)));
    return lit(std::stoll(std::string(token)));
}

static expr parse_func_or_var(sym_parser& p)
{
    char c = p.peek_char();
    if((std::isalpha(c) == 0) and c != '_')
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
    return call_function(sname, args);
}

static expr parse_paren_expr(sym_parser& p)
{
    if(not p.match(std::string_view("(")))
        return {};
    auto e = parse_expr(p);
    p.expect(std::string_view(")"));
    return e;
}

static expr parse_primary(sym_parser& p)
{
    return p.first_of(&parse_paren_expr,
                      &parse_func_or_var,
                      &parse_number,
                      [](sym_parser& q) -> expr { MIGRAPHX_THROW(q.error_message("expression")); });
}

static expr parse_unary(sym_parser& p)
{
    if(p.match(std::string_view("-")))
        return -parse_unary(p);
    return parse_primary(p);
}

static expr parse_mul_expr(sym_parser& p)
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

static expr parse_expr(sym_parser& p)
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

expr parse(const std::string& str)
{
    std::string_view sv(str);
    sym_parser p{sv};
    // skip leading whitespace
    p.advance(0);
    if(p.done())
        return {};
    auto result = parse_expr(p);
    if(not p.done())
        MIGRAPHX_THROW(p.error_message("end of input"));
    return result;
}

static migraphx::value sym_scalar_to_value(const sym::scalar& sv)
{
    return std::visit([](auto x) -> migraphx::value { return migraphx::to_value(x); }, sv);
}

static sym::scalar value_to_sym_scalar(const migraphx::value& v)
{
    if(v.is_float())
        return sym::scalar{v.get_float()};
    return sym::scalar{v.get_int64()};
}

void migraphx_to_value(migraphx::value& v, const sym::interval& i)
{
    migraphx::value result;
    result["min"] = sym_scalar_to_value(i.min);
    result["max"] = sym_scalar_to_value(i.max);
    v             = result;
}

void migraphx_from_value(const migraphx::value& v, sym::interval& i)
{
    i.min = value_to_sym_scalar(v.at("min"));
    i.max = value_to_sym_scalar(v.at("max"));
}

static migraphx::value expr_to_value(const sym::expr& e)
{
    if(e.empty())
        return {};
    migraphx::value result;
    std::visit(
        [&](const auto& n) {
            using t = std::decay_t<decltype(n)>;
            if constexpr(std::is_same<t, sym::literal_node>{})
            {
                result["type"]  = "literal";
                result["value"] = sym_scalar_to_value(n.val);
            }
            else if constexpr(std::is_same<t, sym::variable_node>{})
            {
                result["type"] = "variable";
                result["name"] = n.name;
                if(not n.constraints.empty())
                    result["constraints"] = migraphx::to_value(n.constraints);
                if(not n.optimals.empty())
                {
                    migraphx::value opt_vals;
                    std::transform(n.optimals.begin(),
                                   n.optimals.end(),
                                   std::back_inserter(opt_vals),
                                   [](const scalar& s) { return sym_scalar_to_value(s); });
                    result["optimals"] = opt_vals;
                }
            }
            else
            {
                result["type"] = "op";
                result["name"] = n.op->name;
            }
        },
        get_node(e));
    const auto& children = e.children();
    if(not children.empty())
    {
        std::vector<migraphx::value> child_vals;
        child_vals.reserve(children.size());
        std::transform(children.begin(),
                       children.end(),
                       std::back_inserter(child_vals),
                       [](const sym::expr& c) { return expr_to_value(c); });
        result["children"] = child_vals;
    }
    return result;
}

void migraphx_to_value(migraphx::value& v, const sym::expr& e) { v = expr_to_value(e); }

void migraphx_from_value(const migraphx::value& v, sym::expr& e)
{
    if(v.is_null())
    {
        e = sym::expr{};
        return;
    }
    auto type = v.at("type").get_string();
    if(type == "literal")
    {
        e = sym::lit(value_to_sym_scalar(v.at("value")));
    }
    else if(type == "variable")
    {
        auto name = v.at("name").get_string();
        std::vector<interval> constraints;
        if(v.contains("constraints"))
            constraints = migraphx::from_value<std::vector<interval>>(v.at("constraints"));
        std::set<scalar> optimals;
        if(v.contains("optimals"))
        {
            const auto& opt_vals = v.at("optimals");
            std::transform(opt_vals.begin(),
                           opt_vals.end(),
                           std::inserter(optimals, optimals.end()),
                           [](const migraphx::value& ov) { return value_to_sym_scalar(ov); });
        }
        e = expr(variable_node{std::move(name), std::move(constraints), std::move(optimals)});
    }
    else
    {
        auto name = v.at("name").get_string();
        std::vector<sym::expr> children;
        if(v.contains("children"))
        {
            const auto& cv = v.at("children");
            children.reserve(cv.size());
            std::transform(
                cv.begin(), cv.end(), std::back_inserter(children), [](const migraphx::value& c) {
                    return migraphx::from_value<sym::expr>(c);
                });
        }
        e = sym::call_function(name, children);
    }
}

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
