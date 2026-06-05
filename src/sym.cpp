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
#include <migraphx/hash.hpp>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>
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

// Evaluate f at the four corners of a x b and return the enclosing interval.
template <class F>
static interval corner_extrema(F f, interval a, interval b)
{
    auto p1 = scalar_invoke_common(f, a.min, b.min);
    auto p2 = scalar_invoke_common(f, a.min, b.max);
    auto p3 = scalar_invoke_common(f, a.max, b.min);
    auto p4 = scalar_invoke_common(f, a.max, b.max);
    return {scalar_min(scalar_min(p1, p2), scalar_min(p3, p4)),
            scalar_max(scalar_max(p1, p2), scalar_max(p3, p4))};
}

bool interval::valid() const { return max >= min; }

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
    return corner_extrema([](auto x, auto y) { return x * y; }, a, b);
}

interval operator/(interval a, interval b)
{
    auto b_lo = to<double>(b.min);
    auto b_hi = to<double>(b.max);
    // If the divisor brackets zero, the 4-corner formula is wrong (and may hit
    // integer division-by-zero). Handle it explicitly using infinities so the
    // unbounded regions are representable.
    if(b_lo <= 0.0 and b_hi >= 0.0)
    {
        if(b_lo == 0.0 and b_hi == 0.0)
            MIGRAPHX_THROW("Interval division by zero");
        constexpr double inf = std::numeric_limits<double>::infinity();
        // Strictly crosses zero: 1/b sweeps the full real line.
        if(b_lo < 0.0 and b_hi > 0.0)
            return {-inf, inf};
        auto a_lo = to<double>(a.min);
        auto a_hi = to<double>(a.max);
        // b == [0, b_hi], b_hi > 0: 1/b in [1/b_hi, +inf).
        if(b_lo == 0.0)
        {
            if(a_lo >= 0.0)
                return {a_lo / b_hi, inf};
            if(a_hi <= 0.0)
                return {-inf, a_hi / b_hi};
            return {-inf, inf};
        }
        // b == [b_lo, 0], b_lo < 0: 1/b in (-inf, 1/b_lo].
        if(a_lo >= 0.0)
            return {-inf, a_lo / b_lo};
        if(a_hi <= 0.0)
            return {a_hi / b_lo, inf};
        return {-inf, inf};
    }

    return corner_extrema([](auto x, auto y) { return x / y; }, a, b);
}

interval operator%(interval, interval b)
{
    // The 4-corner min/max formula is wrong for mod (e.g. [1,5] % [3,3] should
    // include {0,1,2}, not just the corner values). Use a loose but correct
    // bound: |a % b| < max(|b_lo|, |b_hi|).
    auto b_lo = to<double>(b.min);
    auto b_hi = to<double>(b.max);
    if(b_lo == 0.0 and b_hi == 0.0)
        MIGRAPHX_THROW("Interval mod by zero");
    auto max_abs = std::max(std::abs(b_lo), std::abs(b_hi));
    if(std::holds_alternative<int64_t>(b.min) and std::holds_alternative<int64_t>(b.max))
    {
        auto m = static_cast<int64_t>(max_abs);
        return {int64_t{-m}, m};
    }
    return {-max_abs, max_abs};
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
    visit([&](auto x) { os << x; }, i.min);
    os << ", ";
    visit([&](auto x) { os << x; }, i.max);
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

interval tan(interval x)
{
    double lo       = to<double>(x.min);
    double hi       = to<double>(x.max);
    const double pi = std::acos(-1.0);
    constexpr double inf = std::numeric_limits<double>::infinity();

    // tan has period pi and poles at pi/2 + k*pi
    if(hi - lo >= pi)
        return {-inf, inf};
    double k = std::ceil((lo - pi / 2.0) / pi);
    if(pi / 2.0 + k * pi <= hi)
        return {-inf, inf};

    double tlo = std::tan(lo);
    double thi = std::tan(hi);
    return {std::min(tlo, thi), std::max(tlo, thi)};
}

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

interval pow(interval x, interval y) { return corner_extrema(MIGRAPHX_LIFT(std::pow), x, y); }

interval min(interval x, interval y)
{
    return {scalar_min(x.min, y.min), scalar_min(x.max, y.max)};
}

interval max(interval x, interval y)
{
    return {scalar_max(x.min, y.min), scalar_max(x.max, y.max)};
}

static std::size_t hash_scalar(scalar s)
{
    return visit(
        [](auto x) -> std::size_t {
            using T = std::decay_t<decltype(x)>;
            if constexpr(std::is_floating_point<T>{})
            {
                int64_t i = x;
                if(float_equal(x, i))
                    return hash_value(i);
            }
            return hash_value(x);
        },
        s);
}

struct literal_node
{
    scalar val;
    std::size_t hash() const { return hash_scalar(val); }
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

    // Identity includes the metadata so equal values are substitutable (regular
    // equality): two variable nodes that differ in constraints or optimals
    // evaluate differently, so they must not compare equal. Use same_symbol /
    // as_symbol to compare ignoring metadata.
    std::size_t hash() const
    {
        std::size_t h = hash_value(name);
        for(const auto& c : constraints)
        {
            hash_combine(h, hash_scalar(c.min));
            hash_combine(h, hash_scalar(c.max));
        }
        for(const auto& o : optimals)
            hash_combine(h, hash_scalar(o));
        return h;
    }
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

    std::size_t hash() const { return hash_value(op->name); }
};

using node_variant = std::variant<literal_node, variable_node, op_node>;

static std::size_t hash_node(const node_variant& nv)
{
    return std::visit([](const auto& x) { return x.hash(); }, nv);
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

static std::string_view get_sym_name(const node_variant& nv)
{
    return std::visit(overloaded{[](const variable_node& n) -> std::string_view { return n.name; },
                                 [](const op_node& n) -> std::string_view { return n.op->name; },
                                 [](const literal_node&) -> std::string_view { return ""; }},
                      nv);
}

static std::string_view get_node_name(const node_variant& nv)
{
    return std::visit(
        overloaded{[](const literal_node&) -> std::string_view { return "literal"; },
                   [](const variable_node&) -> std::string_view { return "variable"; },
                   [](const op_node& n) -> std::string_view { return n.op->name; }},
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
    auto h = hash_node(node);
    hash_range(h, children.begin(), children.end());
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
    if(not constraint.valid())
        MIGRAPHX_THROW("Invalid interval");
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
                                     return pv == std::get<variable_node>(get_node(e));
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

// Structural lexicographic order on (min, max) -- distinct from interval's
// operator< (which is the semantic "strictly below"). Used to keep a
// variable's constraint vector in a canonical sorted+deduped form.
static bool interval_struct_less(const interval& a, const interval& b)
{
    if(scalar_less(a.min, b.min))
        return true;
    if(scalar_less(b.min, a.min))
        return false;
    return scalar_less(a.max, b.max);
}

static bool interval_struct_equal(const interval& a, const interval& b)
{
    return not interval_struct_less(a, b) and not interval_struct_less(b, a);
}

// Canonicalize a constraint set: sort structurally and drop duplicates, so the
// set of assertions has a single representation regardless of insertion order.
// This is what lets variable_node's positional operator==/hash stay correct
// under metadata merging (option B1).
static void normalize_constraints(std::vector<interval>& cs)
{
    std::stable_sort(cs.begin(), cs.end(), interval_struct_less);
    cs.erase(std::unique(cs.begin(), cs.end(), interval_struct_equal), cs.end());
}

// Reduce a variable's constraint set to one effective interval: intersect all
// assertions ([max mins, min maxs]); if that is empty (some pair is disjoint),
// fall back to the convex hull of all ([min mins, max maxs]). Both are computed
// over the whole set at once, so the result does not depend on the set's order.
static interval resolve_constraints(const std::vector<interval>& cs)
{
    scalar imin = cs.front().min;
    scalar imax = cs.front().max;
    scalar hmin = cs.front().min;
    scalar hmax = cs.front().max;
    for(std::size_t i = 1; i < cs.size(); ++i)
    {
        imin = scalar_max(imin, cs[i].min);
        imax = scalar_min(imax, cs[i].max);
        hmin = scalar_min(hmin, cs[i].min);
        hmax = scalar_max(hmax, cs[i].max);
    }
    interval intersection{imin, imax};
    return intersection.valid() ? intersection : interval{hmin, hmax};
}

// The variable's effective interval, or nullopt when it carries no constraints.
static std::optional<interval> variable_interval(const variable_node& n)
{
    if(n.constraints.empty())
        return std::nullopt;
    return resolve_constraints(n.constraints);
}

// Combine the metadata of a group of same-name variable nodes into one by
// taking the union of their constraint sets and the union of their optimals.
// Union is commutative and associative, so the result is order-independent;
// intersect-vs-hull resolution is deferred to read time (resolve_constraints).
static variable_node combine_variables(const std::vector<const variable_node*>& vs)
{
    variable_node r;
    r.name = vs.front()->name;
    for(const auto* v : vs)
    {
        r.constraints.insert(r.constraints.end(), v->constraints.begin(), v->constraints.end());
        r.optimals.insert(v->optimals.begin(), v->optimals.end());
    }
    normalize_constraints(r.constraints);
    return r;
}

// Combine a group of same-symbol exprs (pairwise equal under same_symbol, i.e.
// identical structure differing only in variable metadata) into one, combining
// the metadata of corresponding variable nodes n-ary. Structure is taken from
// the first; recursion is position-wise, so the result is order-independent.
static expr combine_symbols(const std::vector<expr>& group)
{
    const expr& rep = group.front();
    return std::visit(overloaded{[&](const variable_node&) {
                                     std::vector<const variable_node*> vs;
                                     vs.reserve(group.size());
                                     for(const auto& e : group)
                                         vs.push_back(std::get_if<variable_node>(&get_node(e)));
                                     return expr(combine_variables(vs));
                                 },
                                 [&](const literal_node&) { return rep; },
                                 [&](const op_node& o) {
                                     std::vector<expr> children;
                                     children.reserve(rep.children().size());
                                     for(std::size_t i = 0; i < rep.children().size(); ++i)
                                     {
                                         std::vector<expr> column;
                                         column.reserve(group.size());
                                         for(const auto& e : group)
                                             column.push_back(e.children()[i]);
                                         children.push_back(combine_symbols(column));
                                     }
                                     return expr(op_node{o.op}, std::move(children));
                                 }},
                      get_node(rep));
}

static expr normalize_add(const op_def* op, std::vector<expr> args)
{
    std::vector<term> terms;
    terms.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(terms), extract_term);

    std::stable_sort(terms.begin(), terms.end(), [](const term& a, const term& b) {
        return expr_children_less(a.bases, b.bases);
    });

    // Merge adjacent same-symbol terms: sum their coefficients and combine the
    // metadata of each base across the whole group (n-ary, order-independent).
    // Grouping is by same_symbol (metadata-ignoring) to match the name-only sort
    // key, so e.g. x{2,10} + x folds to 2*x{2,10}.
    std::vector<term> merged;
    group_unique(
        terms.begin(),
        terms.end(),
        [&](auto first, auto last) {
            term acc;
            acc.coeff =
                std::accumulate(first, last, scalar{int64_t{0}}, [](scalar c, const term& t) {
                    return scalar_invoke_common([](auto x, auto y) { return x + y; }, c, t.coeff);
                });
            acc.bases.reserve(first->bases.size());
            for(std::size_t i = 0; i < first->bases.size(); ++i)
            {
                std::vector<expr> column;
                for(auto it = first; it != last; ++it)
                    column.push_back(it->bases[i]);
                acc.bases.push_back(combine_symbols(column));
            }
            merged.push_back(std::move(acc));
        },
        [](const term& a, const term& b) {
            return a.bases.size() == b.bases.size() and
                   std::equal(a.bases.begin(), a.bases.end(), b.bases.begin(), same_symbol);
        });

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
        [](const expr& a) {
            const auto* n = std::get_if<literal_node>(&get_node(a));
            assert(n != nullptr); // partitioned to the literal tail above
            return n->val;
        });

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
        const auto& plus_children = it->children();
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
    assert(args.size() == 2); // div is binary
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

    // x / x == 1 (regardless of the variables' metadata)
    if(same_symbol(num, den))
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

std::optional<bool> strict_less(const expr& a, const expr& b, interval default_bounds)
{
    if(a.empty() or b.empty())
        return std::nullopt;

    if(same_symbol(a, b))
        return false;

    // 1. Interval of b - a. eval_interval is already monotonicity-tightened, so the
    //    cross-term correlations between b and a get picked up automatically.
    auto i = (b - a).eval_interval_default(default_bounds);
    if(scalar_less(scalar{int64_t{0}}, i.min))
        return true;
    if(not scalar_less(scalar{int64_t{0}}, i.max))
        return false;
    // 2. b / a vs 1, when a's interval doesn't include zero.
    auto a_int = a.eval_interval_default(default_bounds);
    bool a_pos = scalar_less(scalar{int64_t{0}}, a_int.min);
    bool a_neg = scalar_less(a_int.max, scalar{int64_t{0}});
    if(a_pos or a_neg)
    {
        auto q_int = (b / a).eval_interval_default(default_bounds);
        if(a_pos)
        {
            // a > 0: a < b iff b/a > 1
            if(scalar_less(scalar{int64_t{1}}, q_int.min))
                return true;
            if(not scalar_less(scalar{int64_t{1}}, q_int.max))
                return false;
        }
        else
        {
            // a < 0: a < b iff b/a < 1 (dividing by negative flips)
            if(scalar_less(q_int.max, scalar{int64_t{1}}))
                return true;
            if(not scalar_less(q_int.min, scalar{int64_t{1}}))
                return false;
        }
    }

    return std::nullopt;
}

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

std::string_view expr::name() const
{
    if(empty())
        return "";
    return get_node_name(get_node(*this));
}

bool expr::is_raw() const { return pimpl and pimpl->raw_flag; }

const std::vector<expr>& expr::children() const
{
    static const std::vector<expr> empty_children = {};
    if(empty())
        return empty_children;
    return pimpl->children;
}

// apply signature is (const expr& e, const op_node& op, std::vector<R> args).
// The expr is passed so a custom apply can re-examine the subtree shape (e.g.
// for monotonicity-based interval tightening). The default _auto_apply
// overloads ignore it.

static scalar
generic_eval_auto_apply(const expr&, const op_node& op, const std::vector<scalar>& args)
{
    return op.op->eval(args);
}

static interval
generic_eval_auto_apply(const expr&, const op_node& op, const std::vector<interval>& args)
{
    return op.op->eval_interval(args);
}

static expr generic_eval_auto_apply(const expr&, const op_node& op, const std::vector<expr>& args)
{
    return call_op(op.op, args);
}

template <class R, class Replace, class Apply>
static R generic_eval(const expr& e, const Replace& replace, const Apply& apply)
{
    if(e.empty())
        return {};
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
    return apply(e, std::get<op_node>(get_node(e)), std::move(args));
}

template <class R, class Replace>
static R generic_eval(const expr& e, const Replace& replace)
{
    return generic_eval<R>(e, replace, MIGRAPHX_LIFT(generic_eval_auto_apply));
}

// ---------------------------------------------------------------------------
// Monotonicity-aware interval evaluation.
//
// Plain interval arithmetic loses precision whenever the same variable appears
// in multiple sub-expressions (`h*w - c*h*w`, `(h-1)/2 - (h-1)/4`, ...). For
// the expressions shape analysis throws at us, almost every node is monotone
// in each free variable, and for monotone functions the extrema over a box
// lie exactly at the corners — so two `eval`s at the right corners give the
// *exact* range.
//
// Monotonicity is detected by symbolic differentiation w.r.t. each free var
// and checking the derivative's sign over the variable bounds. The derivative
// is built using doubles (so `d/dh((h-1)/2) = 0.5`, not `0` from integer
// truncation) and evaluated structurally.
// ---------------------------------------------------------------------------

// Symbolic differentiation of `e` w.r.t. `v`. Result expressions use double
// literals so derivatives like `d/dh((h-1)/2)` come out as `0.5` rather than
// `0` from integer truncation. Supports +, *, and / by a literal divisor only;
// throws on other ops, which causes try_monotone_interval to bail.
static expr diff(const expr& e, const expr& v)
{
    if(e.empty() or e.name() == "literal")
        return lit(0.0);
    if(e.name() == "variable")
        return same_symbol(e, v) ? lit(1.0) : lit(0.0);
    if(e.name() == "+")
    {
        const auto& cs = e.children();
        return std::accumulate(
            cs.begin() + 1, cs.end(), diff(cs.front(), v), [&](expr acc, const expr& c) {
                return std::move(acc) + diff(c, v);
            });
    }
    if(e.name() == "*")
    {
        const auto& cs = e.children();
        expr sum       = lit(0.0);
        for(std::size_t i = 0; i < cs.size(); ++i)
        {
            expr term = diff(cs[i], v);
            for(std::size_t j = 0; j < cs.size(); ++j)
                if(j != i)
                    term = std::move(term) * cs[j];
            sum = std::move(sum) + std::move(term);
        }
        return sum;
    }
    if(e.name() == "/")
    {
        const auto& cs = e.children();
        if(cs.size() != 2)
            MIGRAPHX_THROW("diff: / arity");
        if(cs[1].name() != "literal")
            return expr{}; // non-literal divisors would require the quotient rule and break
                           // monotonicity
        const auto& n = std::get<literal_node>(get_node(cs[1]));
        double c      = to<double>(n.val);
        if(c == 0.0)
            return expr{};
        return diff(cs[0], v) * lit(1.0 / c);
    }
    return expr{};
}

// Collect every free variable appearing in `e`, with its effective interval
// (from `lookup` if it resolves the var, otherwise the variable_node's own
// constraint). Throws on any unconstrained variable; that aborts the monotone
// path.
static std::vector<std::pair<expr, interval>>
collect_free_vars(const expr& e, const std::function<std::optional<interval>(const expr&)>& lookup)
{
    std::vector<std::pair<expr, interval>> result;
    std::unordered_set<expr> seen;
    fix([&](auto self, const expr& x) {
        if(x.empty() or not seen.insert(x).second)
            return;
        if(auto iv = lookup(x))
        {
            result.emplace_back(x, *iv);
            return;
        }
        if(x.name() == "variable")
        {
            const auto& n = std::get<variable_node>(get_node(x));
            result.emplace_back(x, variable_interval(n).value_or(interval{}));
            return;
        }
        for(const auto& c : x.children())
            self(c);
    })(e);
    return result;
}

// `lookup` resolves a (variable) subexpression to its interval, or nullopt to
// fall back to the node's own constraint / structural recursion. The cache is a
// per-call memo of interval results, so a subexpression reached through multiple
// parents (including via the monotone-path reentry) is only computed once within
// a single evaluation.
static interval
eval_interval_impl(const expr& e,
                   const std::function<std::optional<interval>(const expr&)>& lookup,
                   std::unordered_map<expr, interval>& cache);

// For each free variable v, compute d(e)/dv and check its sign over v's range;
// if every variable has a definite direction the expression is monotone in
// each one and the extrema are at corners, so two evals give the exact range.
// Derivative intervals go through eval_interval_impl so they hit the cache too.
static std::optional<interval>
try_monotone_interval(const expr& e,
                      const std::function<std::optional<interval>(const expr&)>& lookup,
                      std::unordered_map<expr, interval>& cache)
{
    if(e.empty())
        return std::nullopt;
    auto fvs = collect_free_vars(e, lookup);
    if(fvs.empty())
    {
        auto v = e.eval({});
        return interval{v, v};
    }
    constexpr std::size_t max_vars = 16;
    if(fvs.size() > max_vars)
        return std::nullopt;

    struct mono_info
    {
        expr var;
        interval iv;
        int dir;
    };
    std::vector<mono_info> infos;
    infos.reserve(fvs.size());
    for(const auto& fv : fvs)
    {
        auto deriv = diff(e, fv.first);
        if(deriv.empty())
            return std::nullopt;
        auto di = eval_interval_impl(deriv, lookup, cache);
        // 0 <= min => non-negative derivative => non-decreasing in this var
        bool nonneg = not scalar_less(di.min, scalar{int64_t{0}});
        // max <= 0 => non-positive derivative => non-increasing
        bool nonpos = not scalar_less(scalar{int64_t{0}}, di.max);
        int dir;
        if(nonneg)
            dir = +1;
        else if(nonpos)
            dir = -1;
        else
            return std::nullopt;
        infos.push_back({fv.first, fv.second, dir});
    }

    auto eval_at = [&](bool maxify) {
        std::unordered_map<expr, scalar> point;
        for(const auto& m : infos)
        {
            bool hi      = (m.dir >= 0) == maxify;
            point[m.var] = hi ? m.iv.max : m.iv.min;
        }
        return e.eval(point);
    };
    return interval{eval_at(false), eval_at(true)};
}

// The actual cached evaluator. Cache is keyed on the full subexpression and
// stores the tightened (structural ∩ monotone) interval; the lookup in the
// replace lambda short-circuits the whole subtree walk for nodes already seen.
static interval
eval_interval_impl(const expr& e,
                   const std::function<std::optional<interval>(const expr&)>& lookup,
                   std::unordered_map<expr, interval>& cache)
{
    return generic_eval<interval>(
        e,
        [&](const expr& sub) -> std::optional<interval> {
            if(auto cit = cache.find(sub); cit != cache.end())
                return cit->second;
            if(auto iv = lookup(sub))
            {
                cache.emplace(sub, *iv);
                return iv;
            }
            return std::visit(
                overloaded{[&](const literal_node& n) -> std::optional<interval> {
                               interval r{n.val, n.val};
                               cache.emplace(sub, r);
                               return r;
                           },
                           [&](const variable_node& n) -> std::optional<interval> {
                               auto ci = variable_interval(n);
                               if(not ci)
                                   MIGRAPHX_THROW("Variable '" + n.name +
                                                  "' not found in interval map");
                               cache.emplace(sub, *ci);
                               return ci;
                           },
                           [](const op_node&) -> std::optional<interval> { return std::nullopt; }},
                get_node(sub));
        },
        // Structural interval, intersected with the monotone-corner evaluation
        // when the subtree is monotone in each free variable.
        [&](const expr& sub, const op_node& op, std::vector<interval> args) -> interval {
            auto structural = generic_eval_auto_apply(sub, op, args);
            auto mono       = try_monotone_interval(sub, lookup, cache);
            interval result;
            if(not mono)
            {
                result = structural;
            }
            else
            {
                auto tighter_min = [](scalar s, scalar t) { return scalar_less(s, t) ? t : s; };
                auto tighter_max = [](scalar s, scalar t) { return scalar_less(t, s) ? t : s; };
                result           = {tighter_min(structural.min, mono->min),
                                    tighter_max(structural.max, mono->max)};
            }
            cache.emplace(sub, result);
            return result;
        });
}

// Bottom-up structure-preserving rewrite: f is applied to each node after its
// children have been transformed, and `f` must return the same expr object when
// it leaves a node unchanged. transform_expr returns the original expr (sharing
// pimpl) whenever nothing below changed, so an identity transform allocates
// nothing and an isolated change rebuilds only the spine above it.
//
// max_depth bounds how many levels are visited: the root is depth 1, and nodes
// deeper than max_depth are returned untouched (f is not applied to them).
// A negative max_depth means unlimited.
//
// Op nodes are rebuilt directly (expr(op_node, children)) rather than through
// call_op, so the already-normalized shape is preserved verbatim and we never
// re-enter call_op -> normalize -> eval.
template <class F>
static expr transform_expr(const expr& e, F f, int max_depth = -1)
{
    if(e.empty() or max_depth == 0)
        return e;
    const auto& children = e.children();
    std::vector<expr> new_children;
    new_children.reserve(children.size());
    int child_depth = max_depth < 0 ? -1 : max_depth - 1;
    std::transform(children.begin(),
                   children.end(),
                   std::back_inserter(new_children),
                   [&](const expr& child) { return transform_expr(child, f, child_depth); });
    bool changed = children != new_children;
    expr node    = changed ? expr(std::get<op_node>(get_node(e)), std::move(new_children)) : e;
    return f(node);
}

// Project an expr onto its pure structural symbol form by deep-stripping every
// variable's metadata (constraints, optimals). Exprs that differ only in
// variable metadata share the same as_symbol result, so it is the projection
// used to look up variables regardless of their bounds. A symbol-only expr is
// returned unchanged (sharing pimpl).
//
// max_depth limits the strip to the top max_depth levels (see transform_expr);
// callers that only need to match keys up to a known depth can pass that depth
// to avoid stripping deeper subexpressions that can never match. Default is
// unlimited.
expr as_symbol(const expr& e, int max_depth)
{
    return transform_expr(
        e,
        [](const expr& sub) -> expr {
            const auto* v = std::get_if<variable_node>(&get_node(sub));
            if(v == nullptr or (v->constraints.empty() and v->optimals.empty()))
                return sub;
            return var(v->name);
        },
        max_depth);
}

// Equivalent to as_symbol(a) == as_symbol(b) but compared in lockstep: no
// stripped trees are materialized and the walk short-circuits on the first
// structural mismatch. Variable nodes match on name alone (metadata ignored);
// literals and ops match exactly, then children recurse.
bool same_symbol(const expr& a, const expr& b)
{
    if(a.empty() or b.empty())
        return a.empty() and b.empty();
    const auto& na = get_node(a);
    const auto& nb = get_node(b);
    if(na.index() != nb.index())
        return false;
    bool node_match = std::visit(
        overloaded{
            [&](const literal_node& la) { return la == std::get<literal_node>(nb); },
            [&](const variable_node& va) { return va.name == std::get<variable_node>(nb).name; },
            [&](const op_node& oa) { return oa == std::get<op_node>(nb); }},
        na);
    if(not node_match)
        return false;
    const auto& ca = a.children();
    const auto& cb = b.children();
    return ca.size() == cb.size() and
           std::equal(ca.begin(), ca.end(), cb.begin(), [](const expr& x, const expr& y) {
               return same_symbol(x, y);
           });
}

// Number of levels in e: a leaf (literal/variable) is depth 1, empty is 0.
static int expr_depth(const expr& e)
{
    if(e.empty())
        return 0;
    const auto& children = e.children();
    return 1 + transform_accumulate(
                   children.begin(),
                   children.end(),
                   0,
                   [](int a, int b) { return std::max(a, b); },
                   [](const expr& c) { return expr_depth(c); });
}

// Build a lookup function over a fixed map keyed on exprs that resolves keys
// ignoring variable metadata. The exact (constraint-aware) hash lookup is tried
// first; on a miss the query is projected to its bare-symbol form with as_symbol
// and hash-looked-up again, so an entry keyed on the bare expression resolves a
// query whose variables carry constraints (e.g. key `x + y` matches a query
// `x + y` whose x, y are constrained). Both probes are O(1).
//
// The map's deepest key is computed once up front and passed as the as_symbol
// depth limit: a query that matches a key has that key's depth, so stripping is
// only ever needed in the top max_key_depth levels. A query deeper than every
// key strips only those top levels, stays unequal to all keys, and misses
// harmlessly -- so no per-lookup depth check is needed. The returned closure
// captures the map by reference, so the map must outlive it.
template <class Map>
static auto make_find_symbol(const Map& m)
{
    int max_key_depth = transform_accumulate(
        m.begin(),
        m.end(),
        0,
        [](int a, int b) { return std::max(a, b); },
        [](const auto& kv) { return expr_depth(kv.first); });
    return [&m, max_key_depth](const expr& e) -> const typename Map::mapped_type* {
        if(auto it = m.find(e); it != m.end())
            return &it->second;
        if(e.empty() or max_key_depth == 0)
            return nullptr;
        auto sym = as_symbol(e, max_key_depth);
        if(sym == e)
            return nullptr;
        if(auto it = m.find(sym); it != m.end())
            return &it->second;
        return nullptr;
    };
}

template <class F>
static scalar eval_impl(const expr& root, F lookup)
{
    return generic_eval<scalar>(root, [&](const expr& e) -> std::optional<scalar> {
        if(auto v = lookup(e))
            return *v;
        return std::visit(
            overloaded{[](const literal_node& n) -> std::optional<scalar> { return n.val; },
                       [](const variable_node& n) -> std::optional<scalar> {
                           // A variable whose resolved constraint is fixed
                           // (min == max) has a known value, so use it.
                           auto ci = variable_interval(n);
                           if(ci and ci->min == ci->max)
                               return ci->min;
                           return std::nullopt;
                       },
                       [](const auto&) -> std::optional<scalar> { return std::nullopt; }},
            get_node(e));
    });
}

static bool is_unsigned(scalar x)
{
    return std::holds_alternative<int64_t>(x) and std::get<int64_t>(x) >= 0;
}

std::size_t expr::eval_uint(const std::unordered_map<expr, std::size_t>& symbol_map) const
{
    auto find   = make_find_symbol(symbol_map);
    auto lookup = [&](const expr& sub) -> std::optional<scalar> {
        if(const auto* v = find(sub))
            return scalar(*v);
        return std::nullopt;
    };
    auto r = eval_impl(*this, lookup);
    if(not is_unsigned(r))
        MIGRAPHX_THROW("Result is not an unsigned integer.");
    return to<std::size_t>(r);
}

expr expr::subs(const std::unordered_map<expr, expr>& symbol_map) const
{
    auto find = make_find_symbol(symbol_map);
    return generic_eval<expr>(*this, [&](const expr& e) -> std::optional<expr> {
        if(const auto* v = find(e))
            return *v;
        if(e.empty())
            return e;
        return std::visit(
            overloaded{[&](const literal_node&) -> std::optional<expr> { return e; },
                       [&](const variable_node&) -> std::optional<expr> { return e; },
                       [](const op_node&) -> std::optional<expr> { return std::nullopt; }},
            get_node(e));
    });
}

scalar expr::eval(const std::unordered_map<expr, scalar>& vars) const
{
    auto find   = make_find_symbol(vars);
    auto lookup = [&](const expr& sub) -> std::optional<scalar> {
        if(const auto* v = find(sub))
            return *v;
        return std::nullopt;
    };
    return eval_impl(*this, lookup);
}

interval expr::eval_interval(const std::unordered_map<expr, interval>& vars) const
{
    auto find   = make_find_symbol(vars);
    auto lookup = [&](const expr& sub) -> std::optional<interval> {
        if(const auto* v = find(sub))
            return *v;
        return std::nullopt;
    };
    std::unordered_map<expr, interval> cache;
    return eval_interval_impl(*this, lookup, cache);
}

interval expr::eval_interval_default(interval default_bounds) const
{
    // Resolve every variable to its own constraint, or to default_bounds when
    // it has none, so an unconstrained variable yields the default rather than
    // throwing.
    auto lookup = [&](const expr& sub) -> std::optional<interval> {
        if(sub.empty())
            return std::nullopt;
        const auto* v = std::get_if<variable_node>(&get_node(sub));
        if(v == nullptr)
            return std::nullopt;
        return variable_interval(*v).value_or(default_bounds);
    };
    std::unordered_map<expr, interval> cache;
    return eval_interval_impl(*this, lookup, cache);
}

struct optimal_sample
{
    std::unordered_map<expr, scalar> bindings;
    scalar value;
};

// Combine optimal samples from each child of an op_node.
//
// Each sample carries the variable bindings that produced its value. Children
// are folded in one at a time: for every existing (base, value-list) pair and
// every sample from the next child, the combination is kept only when their
// bindings agree on every shared variable. This makes repeated occurrences of
// the same variable "pair up" (e.g. h*h with h in {2,3} yields {4, 9} rather
// than the cross-product {4, 6, 9}), while subtrees that depend on disjoint
// variables take the full cartesian product. Once all children are folded in,
// the op's eval is applied to each surviving value list.
static std::vector<optimal_sample> combine_optimals(const op_node& op,
                                                    std::vector<std::vector<optimal_sample>> args)
{
    if(args.empty())
        return {{{}, op.op->eval({})}};

    std::vector<std::pair<std::unordered_map<expr, scalar>, std::vector<scalar>>> partial;
    partial.reserve(args.front().size());
    std::transform(args.front().begin(),
                   args.front().end(),
                   std::back_inserter(partial),
                   [](const optimal_sample& s) {
                       return std::make_pair(s.bindings, std::vector<scalar>{s.value});
                   });

    for(std::size_t i = 1; i < args.size(); ++i)
    {
        std::vector<std::pair<std::unordered_map<expr, scalar>, std::vector<scalar>>> next;
        for(const auto& base : partial)
        {
            for(const auto& s : args[i])
            {
                bool compat =
                    std::all_of(s.bindings.begin(), s.bindings.end(), [&](const auto& kv) {
                        auto it = base.first.find(kv.first);
                        return it == base.first.end() or it->second == kv.second;
                    });
                if(not compat)
                    continue;
                auto new_bindings = base.first;
                new_bindings.insert(s.bindings.begin(), s.bindings.end());
                auto new_values = base.second;
                new_values.push_back(s.value);
                next.emplace_back(std::move(new_bindings), std::move(new_values));
            }
        }
        partial = std::move(next);
    }

    std::vector<optimal_sample> result;
    result.reserve(partial.size());
    std::transform(partial.begin(), partial.end(), std::back_inserter(result), [&](auto& p) {
        return optimal_sample{std::move(p.first), op.op->eval(p.second)};
    });
    return result;
}

std::set<scalar> expr::eval_optimals() const
{
    if(empty())
        return {};
    auto samples = generic_eval<std::vector<optimal_sample>>(
        *this,
        [](const expr& e) -> std::optional<std::vector<optimal_sample>> {
            return std::visit(
                overloaded{
                    [](const literal_node& n) -> std::optional<std::vector<optimal_sample>> {
                        return std::vector<optimal_sample>{{{}, n.val}};
                    },
                    [&](const variable_node& n) -> std::optional<std::vector<optimal_sample>> {
                        // No optimals to sample: yield an empty sample set, which
                        // combine_optimals propagates so the whole expression
                        // evaluates to an empty result rather than throwing.
                        if(n.optimals.empty())
                            return std::vector<optimal_sample>{};
                        std::vector<optimal_sample> samples;
                        samples.reserve(n.optimals.size());
                        std::transform(
                            n.optimals.begin(),
                            n.optimals.end(),
                            std::back_inserter(samples),
                            [&](const scalar& v) { return optimal_sample{{{e, v}}, v}; });
                        return samples;
                    },
                    [](const op_node&) -> std::optional<std::vector<optimal_sample>> {
                        return std::nullopt;
                    }},
                get_node(e));
        },
        [](const expr&, const op_node& op, std::vector<std::vector<optimal_sample>> args) {
            return combine_optimals(op, std::move(args));
        });
    std::set<scalar> result;
    std::transform(samples.begin(),
                   samples.end(),
                   std::inserter(result, result.end()),
                   [](const auto& s) { return s.value; });
    return result;
}

std::set<std::size_t> expr::eval_optimals_uint() const
{
    std::set<std::size_t> result;
    auto r = eval_optimals();
    std::transform(r.begin(), r.end(), std::inserter(result, result.end()), [](const scalar& v) {
        if(not is_unsigned(v))
            MIGRAPHX_THROW("Result is not an unsigned integer.");
        return to<std::size_t>(v);
    });
    return result;
}

static std::string scalar_to_string(const scalar& v)
{
    return visit(
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
               [](const expr&, const op_node& op, std::vector<string_prec> args) -> string_prec {
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
    assert(op_n != nullptr); // a non-leaf expr is always an op node
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
    auto token    = p.parse_while([](unsigned char c) { return std::isdigit(c) or c == '.'; });
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
    auto name = p.parse_while([](unsigned char ch) { return std::isalnum(ch) or ch == '_'; });
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
    // A msgpack round trip can re-tag a non-negative integer as uint64, so the
    // unsigned case must be handled explicitly; scalar only holds int64/double,
    // so the uint64 is clamped through scalar's pick_scalar conversion.
    if(v.is_float())
        return sym::scalar{v.get_float()};
    if(const auto* u = v.if_uint64())
        return sym::scalar(*u);
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
        {
            constraints = migraphx::from_value<std::vector<interval>>(v.at("constraints"));
            // Keep the canonical sorted+deduped form so a deserialized node
            // compares equal to an equivalently-constructed one (option B1).
            normalize_constraints(constraints);
        }
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

namespace std {
std::size_t hash<migraphx::sym::expr>::operator()(const migraphx::sym::expr& e) const
{
    return e.hash();
}
} // namespace std
