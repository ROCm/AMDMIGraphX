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
#include <migraphx/functional.hpp>
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ===================================================================
// Section 1: Expression node types
// ===================================================================

struct expr_node;
using expr_ptr = std::shared_ptr<const expr_node>;

struct expr_compare
{
    bool operator()(const expr_ptr& a, const expr_ptr& b) const;
};

using term_map   = std::map<expr_ptr, int64_t, expr_compare>;
using factor_map = std::map<expr_ptr, int64_t, expr_compare>;

struct integer_data
{
    int64_t value;
};
struct symbol_data
{
    std::string name;
    int64_t min;
    int64_t max;
};
struct add_data
{
    int64_t constant;
    term_map terms;
};
struct mul_data
{
    int64_t coefficient;
    factor_map factors;
};
struct tdiv_data
{
    expr_ptr numerator;
    expr_ptr denominator;
};

using expr_data = std::variant<integer_data, symbol_data, add_data, mul_data, tdiv_data>;

struct expr_node
{
    expr_data data;
    std::size_t cached_hash = 0;
};

template <class T>
static bool holds(const expr_ptr& e)
{
    return std::holds_alternative<T>(e->data);
}

static int64_t get_integer(const expr_ptr& e) { return std::get<integer_data>(e->data).value; }
static const add_data& get_add(const expr_ptr& e) { return std::get<add_data>(e->data); }
static const mul_data& get_mul(const expr_ptr& e) { return std::get<mul_data>(e->data); }

// ===================================================================
// Section 2: Hash computation
// ===================================================================

static std::size_t hash_combine(std::size_t seed, std::size_t v)
{
    return seed ^ (v + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

template <class Map>
static std::size_t hash_ordered_map(const Map& m)
{
    std::size_t h = 0;
    for(const auto& [key, val] : m)
    {
        h = hash_combine(h, key->cached_hash);
        h = hash_combine(h, std::hash<int64_t>{}(val));
    }
    return h;
}

static std::size_t compute_hash(const expr_data& d)
{
    std::size_t h = std::hash<std::size_t>{}(d.index());
    return std::visit(
        overloaded{
            [&](const integer_data& p) { return hash_combine(h, std::hash<int64_t>{}(p.value)); },
            [&](const symbol_data& p) {
                auto h2 = hash_combine(h, std::hash<std::string>{}(p.name));
                h2      = hash_combine(h2, std::hash<int64_t>{}(p.min));
                return hash_combine(h2, std::hash<int64_t>{}(p.max));
            },
            [&](const add_data& p) {
                return hash_combine(hash_combine(h, std::hash<int64_t>{}(p.constant)),
                                    hash_ordered_map(p.terms));
            },
            [&](const mul_data& p) {
                return hash_combine(hash_combine(h, std::hash<int64_t>{}(p.coefficient)),
                                    hash_ordered_map(p.factors));
            },
            [&](const tdiv_data& p) {
                return hash_combine(hash_combine(h, p.numerator->cached_hash),
                                    p.denominator->cached_hash);
            }},
        d);
}

// ===================================================================
// Section 3: Canonical ordering (expr_compare)
// ===================================================================

static int compare_expr(const expr_ptr& a, const expr_ptr& b);

template <class Map>
static int compare_maps(const Map& a, const Map& b)
{
    auto it_a = a.begin();
    auto it_b = b.begin();
    for(; it_a != a.end() and it_b != b.end(); ++it_a, ++it_b)
    {
        int c = compare_expr(it_a->first, it_b->first);
        if(c != 0)
            return c;
        if(it_a->second < it_b->second)
            return -1;
        if(it_a->second > it_b->second)
            return 1;
    }
    if(it_a != a.end())
        return 1;
    if(it_b != b.end())
        return -1;
    return 0;
}

static int compare_expr(const expr_ptr& a, const expr_ptr& b)
{
    if(a->data.index() != b->data.index())
        return a->data.index() < b->data.index() ? -1 : 1;

    return std::visit(
        overloaded{[&](const integer_data& da) {
                       const auto& db = std::get<integer_data>(b->data);
                       return (da.value < db.value) ? -1 : (da.value > db.value) ? 1 : 0;
                   },
                   [&](const symbol_data& da) {
                       const auto& db = std::get<symbol_data>(b->data);
                       int c          = da.name.compare(db.name);
                       if(c != 0)
                           return c;
                       if(da.min != db.min)
                           return da.min < db.min ? -1 : 1;
                       if(da.max != db.max)
                           return da.max < db.max ? -1 : 1;
                       return 0;
                   },
                   [&](const add_data& da) {
                       const auto& db = std::get<add_data>(b->data);
                       if(da.constant != db.constant)
                           return da.constant < db.constant ? -1 : 1;
                       return compare_maps(da.terms, db.terms);
                   },
                   [&](const mul_data& da) {
                       const auto& db = std::get<mul_data>(b->data);
                       if(da.coefficient != db.coefficient)
                           return da.coefficient < db.coefficient ? -1 : 1;
                       return compare_maps(da.factors, db.factors);
                   },
                   [&](const tdiv_data& da) {
                       const auto& db = std::get<tdiv_data>(b->data);
                       int c          = compare_expr(da.numerator, db.numerator);
                       if(c != 0)
                           return c;
                       return compare_expr(da.denominator, db.denominator);
                   }},
        a->data);
}

bool expr_compare::operator()(const expr_ptr& a, const expr_ptr& b) const
{
    return compare_expr(a, b) < 0;
}

// ===================================================================
// Section 4: Structural equality
// ===================================================================

static bool expr_equal(const expr_ptr& a, const expr_ptr& b)
{
    if(a.get() == b.get())
        return true;
    if(a->cached_hash != b->cached_hash)
        return false;
    return compare_expr(a, b) == 0;
}

// ===================================================================
// Section 5: Factory functions (canonical constructors)
// ===================================================================

static expr_ptr make_node(expr_data d)
{
    auto n         = std::make_shared<expr_node>();
    n->data        = std::move(d);
    n->cached_hash = compute_hash(n->data);
    return n;
}

static const expr_ptr& const_zero()
{
    static auto p = make_node(integer_data{0});
    return p;
}
static const expr_ptr& const_one()
{
    static auto p = make_node(integer_data{1});
    return p;
}
static const expr_ptr& const_neg_one()
{
    static auto p = make_node(integer_data{-1});
    return p;
}

static expr_ptr make_integer(int64_t n)
{
    if(n == 0)
        return const_zero();
    if(n == 1)
        return const_one();
    if(n == -1)
        return const_neg_one();
    return make_node(integer_data{n});
}

static expr_ptr make_symbol(const std::string& name, int64_t min, int64_t max)
{
    return make_node(symbol_data{name, min, max});
}

static expr_ptr make_add(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_sub(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_neg(const expr_ptr& a);
static expr_ptr make_mul(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_trunc_div(const expr_ptr& a, const expr_ptr& b);
static expr_ptr build_mul(int64_t coefficient, factor_map factors);

struct add_parts
{
    int64_t constant = 0;
    term_map terms;
};

static add_parts extract_add(const expr_ptr& e)
{
    return std::visit(
        overloaded{[](const integer_data& d) -> add_parts { return {d.value, {}}; },
                   [](const add_data& d) -> add_parts { return {d.constant, d.terms}; },
                   [&](const mul_data& d) -> add_parts {
                       auto base = build_mul(1, d.factors);
                       return {0, {{base, d.coefficient}}};
                   },
                   [&](const auto&) -> add_parts { return {0, {{e, 1}}}; }},
        e->data);
}

static expr_ptr build_add(int64_t constant, term_map terms)
{
    // Remove zero-coefficient terms
    for(auto it = terms.begin(); it != terms.end();)
    {
        if(it->second == 0)
            it = terms.erase(it);
        else
            ++it;
    }
    if(terms.empty())
        return make_integer(constant);
    if(constant == 0 and terms.size() == 1)
    {
        auto& [term, coeff] = *terms.begin();
        if(coeff == 1)
            return term;
        return make_mul(make_integer(coeff), term);
    }
    return make_node(add_data{constant, std::move(terms)});
}

static expr_ptr make_add(const expr_ptr& a, const expr_ptr& b)
{
    auto pa = extract_add(a);
    auto pb = extract_add(b);

    int64_t constant = pa.constant + pb.constant;
    term_map terms   = std::move(pa.terms);
    for(const auto& [term, coeff] : pb.terms)
        terms[term] += coeff;

    return build_add(constant, std::move(terms));
}

static expr_ptr make_neg(const expr_ptr& a)
{
    return std::visit(
        overloaded{
            [](const integer_data& d) -> expr_ptr { return make_integer(-d.value); },
            [](const add_data& d) -> expr_ptr {
                term_map negated;
                for(const auto& [term, coeff] : d.terms)
                    negated[term] = -coeff;
                return build_add(-d.constant, std::move(negated));
            },
            [](const mul_data& d) -> expr_ptr { return build_mul(-d.coefficient, d.factors); },
            [&](const auto&) -> expr_ptr { return make_mul(make_integer(-1), a); }},
        a->data);
}

static expr_ptr make_sub(const expr_ptr& a, const expr_ptr& b) { return make_add(a, make_neg(b)); }

struct mul_parts
{
    int64_t coefficient = 1;
    factor_map factors;
};

static mul_parts extract_mul(const expr_ptr& e)
{
    return std::visit(
        overloaded{[](const integer_data& d) -> mul_parts { return {d.value, {}}; },
                   [](const mul_data& d) -> mul_parts { return {d.coefficient, d.factors}; },
                   [&](const auto&) -> mul_parts { return {1, {{e, 1}}}; }},
        e->data);
}

static expr_ptr build_mul(int64_t coefficient, factor_map factors)
{
    if(coefficient == 0)
        return make_integer(0);
    for(auto it = factors.begin(); it != factors.end();)
    {
        if(it->second == 0)
            it = factors.erase(it);
        else
            ++it;
    }
    if(factors.empty())
        return make_integer(coefficient);
    if(coefficient == 1 and factors.size() == 1)
    {
        auto& [base, exp] = *factors.begin();
        if(exp == 1)
            return base;
    }
    return make_node(mul_data{coefficient, std::move(factors)});
}

static expr_ptr make_mul(const expr_ptr& a, const expr_ptr& b)
{
    if(holds<integer_data>(a) and holds<add_data>(b))
    {
        int64_t n = get_integer(a);
        if(n == 0)
            return make_integer(0);
        if(n == 1)
            return b;
        const auto& d = get_add(b);
        term_map scaled;
        for(const auto& [term, coeff] : d.terms)
            scaled[term] = coeff * n;
        return build_add(d.constant * n, std::move(scaled));
    }
    if(holds<integer_data>(b) and holds<add_data>(a))
        return make_mul(b, a);

    auto pa = extract_mul(a);
    auto pb = extract_mul(b);

    int64_t coefficient = pa.coefficient * pb.coefficient;
    if(coefficient == 0)
        return make_integer(0);

    factor_map factors = std::move(pa.factors);
    for(const auto& [base, exp] : pb.factors)
        factors[base] += exp;

    return build_mul(coefficient, std::move(factors));
}

static expr_ptr try_cancel_single(const mul_data& da, const expr_ptr& b)
{
    auto it = da.factors.find(b);
    if(it == da.factors.end())
        return nullptr;
    factor_map reduced = da.factors;
    if(it->second == 1)
        reduced.erase(it->first);
    else
        reduced[it->first] = it->second - 1;
    return build_mul(da.coefficient, std::move(reduced));
}

static expr_ptr try_div_int_over_add(const add_data& d, int64_t den)
{
    if(d.constant % den != 0)
        return nullptr;
    bool all_divisible = std::all_of(
        d.terms.begin(), d.terms.end(), [&](const auto& p) { return p.second % den == 0; });
    if(not all_divisible)
        return nullptr;
    term_map divided = d.terms;
    for(auto& [base, coeff] : divided)
        coeff /= den;
    return build_add(d.constant / den, std::move(divided));
}

static expr_ptr
try_cancel_factors(const mul_data& da, const mul_data& db, const expr_ptr& a, const expr_ptr& b)
{
    factor_map reduced_num = da.factors;
    factor_map reduced_den = db.factors;
    for(auto it_den = reduced_den.begin(); it_den != reduced_den.end();)
    {
        auto it_num = reduced_num.find(it_den->first);
        if(it_num == reduced_num.end())
        {
            ++it_den;
            continue;
        }
        int64_t cancel = std::min(it_num->second, it_den->second);
        it_num->second -= cancel;
        it_den->second -= cancel;
        if(it_num->second == 0)
            reduced_num.erase(it_num);
        if(it_den->second == 0)
            it_den = reduced_den.erase(it_den);
        else
            ++it_den;
    }
    auto new_num = build_mul(da.coefficient, std::move(reduced_num));
    auto new_den = build_mul(db.coefficient, std::move(reduced_den));
    if(not expr_equal(new_num, a) or not expr_equal(new_den, b))
        return make_trunc_div(new_num, new_den);
    return nullptr;
}

static expr_ptr make_trunc_div(const expr_ptr& a, const expr_ptr& b)
{
    if(holds<integer_data>(a) and get_integer(a) == 0)
        return a;

    if(expr_equal(a, b))
        return make_integer(1);

    if(holds<integer_data>(b))
    {
        int64_t den = get_integer(b);
        if(den == 0)
            MIGRAPHX_THROW("symbolic: division by zero");
        if(den == 1)
            return a;
        if(holds<integer_data>(a))
            return make_integer(get_integer(a) / den);
        if(holds<mul_data>(a))
        {
            const auto& d = get_mul(a);
            if(d.coefficient % den == 0)
                return build_mul(d.coefficient / den, d.factors);
        }
        if(holds<add_data>(a))
        {
            auto r = try_div_int_over_add(get_add(a), den);
            if(r != nullptr)
                return r;
        }
    }

    if(holds<mul_data>(a))
    {
        const auto& da = get_mul(a);
        if(holds<mul_data>(b))
        {
            auto r = try_cancel_factors(da, get_mul(b), a, b);
            if(r != nullptr)
                return r;
        }
        else
        {
            auto r = try_cancel_single(da, b);
            if(r != nullptr)
                return r;
        }
    }

    return make_node(tdiv_data{a, b});
}

// ===================================================================
// Section 6: Substitution and evaluation
// ===================================================================

using binding_map = std::map<expr_ptr, int64_t, expr_compare>;
using subs_map    = std::map<expr_ptr, expr_ptr, expr_compare>;

static expr_ptr substitute(const expr_ptr& e, const subs_map& bindings)
{
    return std::visit(overloaded{[&](const integer_data&) -> expr_ptr { return e; },
                                 [&](const symbol_data&) -> expr_ptr {
                                     auto it = bindings.find(e);
                                     if(it != bindings.end())
                                         return it->second;
                                     return e;
                                 },
                                 [&](const add_data& d) -> expr_ptr {
                                     expr_ptr result = make_integer(d.constant);
                                     for(const auto& [term, coeff] : d.terms)
                                     {
                                         auto st = substitute(term, bindings);
                                         result =
                                             make_add(result, make_mul(make_integer(coeff), st));
                                     }
                                     return result;
                                 },
                                 [&](const mul_data& d) -> expr_ptr {
                                     expr_ptr result = make_integer(d.coefficient);
                                     for(const auto& [base, exp] : d.factors)
                                     {
                                         auto sb = substitute(base, bindings);
                                         for(int64_t i = 0; i < exp; ++i)
                                             result = make_mul(result, sb);
                                     }
                                     return result;
                                 },
                                 [&](const tdiv_data& d) -> expr_ptr {
                                     auto sn = substitute(d.numerator, bindings);
                                     auto sd = substitute(d.denominator, bindings);
                                     return make_trunc_div(sn, sd);
                                 }},
                      e->data);
}

template <class SymbolResolver>
static int64_t eval_impl(const expr_ptr& e, const SymbolResolver& resolve_sym)
{
    return std::visit(overloaded{[](const integer_data& d) -> int64_t { return d.value; },
                                 [&](const symbol_data& d) -> int64_t { return resolve_sym(e, d); },
                                 [&](const add_data& d) -> int64_t {
                                     int64_t sum = d.constant;
                                     for(const auto& [term, coeff] : d.terms)
                                         sum += coeff * eval_impl(term, resolve_sym);
                                     return sum;
                                 },
                                 [&](const mul_data& d) -> int64_t {
                                     int64_t prod = d.coefficient;
                                     for(const auto& [base, exp] : d.factors)
                                     {
                                         int64_t val = eval_impl(base, resolve_sym);
                                         for(int64_t i = 0; i < exp; ++i)
                                             prod *= val;
                                     }
                                     return prod;
                                 },
                                 [&](const tdiv_data& d) -> int64_t {
                                     auto denom = eval_impl(d.denominator, resolve_sym);
                                     if(denom == 0)
                                         MIGRAPHX_THROW("sym::expr: division by zero during eval");
                                     return eval_impl(d.numerator, resolve_sym) / denom;
                                 }},
                      e->data);
}

static int64_t eval_direct(const expr_ptr& e, const binding_map& bindings)
{
    return eval_impl(e, [&](const expr_ptr& node, const symbol_data& d) -> int64_t {
        auto it = bindings.find(node);
        if(it != bindings.end())
            return it->second;
        MIGRAPHX_THROW("sym::expr::eval_uint: unbound symbol '" + d.name + "'");
    });
}

static int64_t eval_at_min(const expr_ptr& e)
{
    return eval_impl(e, [](const expr_ptr&, const symbol_data& d) -> int64_t { return d.min; });
}

static int64_t eval_at_max(const expr_ptr& e)
{
    return eval_impl(e, [](const expr_ptr&, const symbol_data& d) -> int64_t { return d.max; });
}

// ===================================================================
// Section 7: Pretty-printer
// ===================================================================

enum
{
    prec_atom = 100,
    prec_mul  = 50,
    prec_add  = 40
};

static std::string print_expr(const expr_ptr& e, int parent_prec = 0);

static std::string print_add(const add_data& d, int parent_prec)
{
    std::ostringstream os;
    bool first = true;
    for(const auto& [term, coeff] : d.terms)
    {
        if(first)
        {
            if(coeff == -1)
                os << "-" << print_expr(term, prec_add);
            else if(coeff == 1)
                os << print_expr(term, prec_add);
            else
                os << coeff << "*" << print_expr(term, prec_mul + 1);
            first = false;
        }
        else
        {
            if(coeff == 1)
                os << " + " << print_expr(term, prec_add);
            else if(coeff == -1)
                os << " - " << print_expr(term, prec_add);
            else if(coeff > 0)
                os << " + " << coeff << "*" << print_expr(term, prec_mul + 1);
            else
                os << " - " << (-coeff) << "*" << print_expr(term, prec_mul + 1);
        }
    }
    if(d.constant > 0)
        os << " + " << d.constant;
    else if(d.constant < 0)
        os << " - " << (-d.constant);
    std::string s = os.str();
    if(parent_prec > prec_add)
        return "(" + s + ")";
    return s;
}

static std::string print_mul(const mul_data& d, int parent_prec)
{
    std::ostringstream os;
    bool first = true;
    if(d.coefficient == -1)
    {
        os << "-";
    }
    else if(d.coefficient != 1)
    {
        os << d.coefficient;
        first = false;
    }
    for(const auto& [base, exp] : d.factors)
    {
        for(int64_t i = 0; i < exp; ++i)
        {
            if(not first)
                os << "*";
            os << print_expr(base, prec_mul + 1);
            first = false;
        }
    }
    std::string raw = os.str();
    if(parent_prec > prec_mul)
        return "(" + raw + ")";
    return raw;
}

static std::string print_expr(const expr_ptr& e, int parent_prec)
{
    return std::visit(
        overloaded{[](const integer_data& d) -> std::string { return std::to_string(d.value); },
                   [](const symbol_data& d) -> std::string { return d.name; },
                   [&](const add_data& d) -> std::string { return print_add(d, parent_prec); },
                   [&](const mul_data& d) -> std::string { return print_mul(d, parent_prec); },
                   [&](const tdiv_data& d) -> std::string {
                       std::string s = print_expr(d.numerator, prec_mul + 1) + "/" +
                                       print_expr(d.denominator, prec_mul + 1);
                       if(parent_prec > prec_mul)
                           return "(" + s + ")";
                       return s;
                   }},
        e->data);
}

// ===================================================================
// Section 8: Recursive descent parser
// ===================================================================

static void skip_ws(const char*& p)
{
    while(*p != '\0' and std::isspace(static_cast<unsigned char>(*p)) != 0)
        ++p;
}

static expr_ptr parse_expr(const char*& p);
static expr_ptr parse_term(const char*& p);
static expr_ptr parse_unary(const char*& p);
static expr_ptr parse_primary(const char*& p);

static expr_ptr parse_primary(const char*& p)
{
    skip_ws(p);
    if(std::isdigit(static_cast<unsigned char>(*p)) != 0)
    {
        int64_t n = 0;
        while(std::isdigit(static_cast<unsigned char>(*p)) != 0)
        {
            n = n * 10 + (*p - '0');
            ++p;
        }
        return make_integer(n);
    }
    if(std::isalpha(static_cast<unsigned char>(*p)) != 0 or *p == '_')
    {
        std::string name;
        while(std::isalnum(static_cast<unsigned char>(*p)) != 0 or *p == '_')
        {
            name += *p;
            ++p;
        }
        return make_symbol(name, 1, 1);
    }
    if(*p == '(')
    {
        ++p;
        auto inner = parse_expr(p);
        skip_ws(p);
        if(*p != ')')
            MIGRAPHX_THROW("symbolic parser: expected ')'");
        ++p;
        return inner;
    }
    MIGRAPHX_THROW("symbolic parser: unexpected character '" + std::string(1, *p) + "'");
}

static expr_ptr parse_unary(const char*& p)
{
    skip_ws(p);
    if(*p == '-')
    {
        ++p;
        return make_neg(parse_unary(p));
    }
    return parse_primary(p);
}

static expr_ptr parse_term(const char*& p)
{
    auto left = parse_unary(p);
    for(;;)
    {
        skip_ws(p);
        if(*p == '*')
        {
            ++p;
            left = make_mul(left, parse_unary(p));
        }
        else if(*p == '/')
        {
            ++p;
            left = make_trunc_div(left, parse_unary(p));
        }
        else
            break;
    }
    return left;
}

static expr_ptr parse_expr(const char*& p)
{
    auto left = parse_term(p);
    for(;;)
    {
        skip_ws(p);
        if(*p == '+')
        {
            ++p;
            left = make_add(left, parse_term(p));
        }
        else if(*p == '-')
        {
            ++p;
            left = make_sub(left, parse_term(p));
        }
        else
            break;
    }
    return left;
}

static expr_ptr parse_string(const std::string& s)
{
    const char* p = s.c_str();
    auto result   = parse_expr(p);
    skip_ws(p);
    if(*p != '\0')
        MIGRAPHX_THROW("symbolic parser: unexpected trailing characters: '" + std::string(p) + "'");
    return result;
}

// ===================================================================
// Section 9: sym::expr public API wrapper
// ===================================================================

namespace sym {

struct expr::impl
{
    expr_ptr node;

    impl() : node(make_integer(0)) {}
    explicit impl(expr_ptr e) : node(std::move(e)) {}
};

expr::expr() = default;

expr::expr(std::shared_ptr<const impl> pi) : p(std::move(pi)) {}

bool expr::empty() const { return p == nullptr; }

std::size_t expr::hash() const
{
    if(empty())
        return 0;
    return p->node->cached_hash;
}

std::string expr::to_string() const
{
    if(empty())
        return {};
    return print_expr(p->node);
}

std::size_t expr::eval_uint(const std::unordered_map<expr, std::size_t>& symbol_map) const
{
    if(empty())
        return 0;
    binding_map bindings;
    for(const auto& [k, v] : symbol_map)
    {
        if(k.empty() or not holds<symbol_data>(k.p->node))
            MIGRAPHX_THROW("sym::expr::eval_uint: map key '" + k.to_string() + "' is not a symbol");
        bindings[k.p->node] = v;
    }
    auto v = eval_direct(p->node, bindings);
    if(v < 0)
        MIGRAPHX_THROW("sym::expr::eval_uint: expression evaluated to negative value");
    return v;
}

expr expr::subs(const std::unordered_map<expr, expr>& symbol_map) const
{
    if(empty())
        return {};
    subs_map bindings;
    for(const auto& [k, v] : symbol_map)
    {
        if(k.empty() or not holds<symbol_data>(k.p->node))
            MIGRAPHX_THROW("sym::expr::subs: map key '" + k.to_string() + "' is not a symbol");
        if(v.empty())
            MIGRAPHX_THROW("sym::expr::subs: substitution value must not be empty");
        bindings[k.p->node] = v.p->node;
    }
    return {std::make_shared<impl>(substitute(p->node, bindings))};
}

expr operator+(const expr& a, const expr& b)
{
    if(a.empty() or b.empty())
        return {};
    return {std::make_shared<expr::impl>(make_add(a.p->node, b.p->node))};
}

expr operator-(const expr& a, const expr& b)
{
    if(a.empty() or b.empty())
        return {};
    return {std::make_shared<expr::impl>(make_sub(a.p->node, b.p->node))};
}

expr operator*(const expr& a, const expr& b)
{
    if(a.empty() or b.empty())
        return {};
    return {std::make_shared<expr::impl>(make_mul(a.p->node, b.p->node))};
}

expr operator/(const expr& a, const expr& b)
{
    if(a.empty() or b.empty())
        return {};
    return {std::make_shared<expr::impl>(make_trunc_div(a.p->node, b.p->node))};
}

bool operator==(const expr& a, const expr& b)
{
    if(a.empty() and b.empty())
        return true;
    if(a.empty() != b.empty())
        return false;
    return expr_equal(a.p->node, b.p->node);
}

bool operator!=(const expr& a, const expr& b) { return not(a == b); }

// Semantic less-than for symbolic expressions using interval arithmetic.
//
// Assumptions:
//   - All symbols have positive intervals [min, max] where 1 <= min <= max.
//   - Expressions are monotonically non-decreasing in each variable, which
//     holds for dimension/stride arithmetic (sums and products of positive
//     terms). This lets us bound the range of (b - a) by evaluating at the
//     interval endpoints.
//
// Algorithm:
//   Compute diff = b - a, then evaluate diff at the lower and upper bounds
//   of every symbol to obtain [lo, hi].  If the entire interval is positive
//   (lo >= 0 and hi > 0) then a < b.  If the interval is non-positive
//   (hi <= 0) then a >= b.  Otherwise the comparison is undetermined and we
//   throw — this indicates a case outside the monotonic/positive assumption.
//
// Examples (all symbols default to [1, 1]):
//   n < 2*n  =>  diff = 2n - n = n,  lo = 1, hi = 1  =>  true
//   2*n < n  =>  diff = n - 2n = -n, lo = -1, hi = -1 => false
//   k < m*k  =>  diff = mk - k = k(m-1), lo = 0, hi = 0 => false (not strictly positive)
//
// With explicit bounds, e.g. n in [2, 10]:
//   n < 3    =>  diff = 3 - n,  lo = 3-10 = -7, hi = 3-2 = 1 => undetermined (throws)
//   n < 11   =>  diff = 11 - n, lo = 11-10 = 1, hi = 11-2 = 9 => true
bool operator<(const expr& a, const expr& b)
{
    if(a.empty() and b.empty())
        return false;
    if(a.empty() or b.empty())
        MIGRAPHX_THROW("sym::expr: cannot compare empty expression");
    auto diff   = make_sub(b.p->node, a.p->node);
    auto val_lo = eval_at_min(diff);
    auto val_hi = eval_at_max(diff);
    auto lo     = std::min(val_lo, val_hi);
    auto hi     = std::max(val_lo, val_hi);
    if(lo >= 0 and hi > 0)
        return true;
    if(hi <= 0)
        return false;
    MIGRAPHX_THROW("sym::expr: comparison undetermined for: " + print_expr(a.p->node) + " < " +
                   print_expr(b.p->node));
}

std::ostream& operator<<(std::ostream& os, const expr& e)
{
    if(not e.empty())
        os << print_expr(e.p->node);
    return os;
}

expr var(const std::string& name, int64_t min, int64_t max)
{
    if(name.empty())
        MIGRAPHX_THROW("sym::var: variable name must not be empty");
    return {std::make_shared<expr::impl>(make_symbol(name, min, max))};
}

expr lit(int64_t n) { return {std::make_shared<expr::impl>(make_integer(n))}; }

expr parse(const std::string& s)
{
    if(s.find_first_not_of(" \t\n\r") == std::string::npos)
        return {};
    return {std::make_shared<expr::impl>(parse_string(s))};
}

static value node_to_value(const expr_ptr& e)
{
    return std::visit(overloaded{[](const integer_data& d) -> value {
                                     value r;
                                     r["type"]  = "int";
                                     r["value"] = d.value;
                                     return r;
                                 },
                                 [](const symbol_data& d) -> value {
                                     value r;
                                     r["type"] = "sym";
                                     r["name"] = d.name;
                                     r["min"]  = d.min;
                                     r["max"]  = d.max;
                                     return r;
                                 },
                                 [](const add_data& d) -> value {
                                     value r;
                                     r["type"]     = "add";
                                     r["constant"] = d.constant;
                                     value terms   = value::array{};
                                     for(const auto& [term, coeff] : d.terms)
                                     {
                                         value t;
                                         t["expr"]  = node_to_value(term);
                                         t["coeff"] = coeff;
                                         terms.push_back(t);
                                     }
                                     r["terms"] = terms;
                                     return r;
                                 },
                                 [](const mul_data& d) -> value {
                                     value r;
                                     r["type"]     = "mul";
                                     r["coeff"]    = d.coefficient;
                                     value factors = value::array{};
                                     for(const auto& [base, exp] : d.factors)
                                     {
                                         value f;
                                         f["expr"] = node_to_value(base);
                                         f["exp"]  = exp;
                                         factors.push_back(f);
                                     }
                                     r["factors"] = factors;
                                     return r;
                                 },
                                 [](const tdiv_data& d) -> value {
                                     value r;
                                     r["type"] = "tdiv";
                                     r["num"]  = node_to_value(d.numerator);
                                     r["den"]  = node_to_value(d.denominator);
                                     return r;
                                 }},
                      e->data);
}

static expr_ptr node_from_value(const value& v)
{
    const auto& type = v.at("type").get_string();
    if(type == "int")
    {
        return make_integer(v.at("value").to<int64_t>());
    }
    else if(type == "sym")
    {
        auto sym_min = v.contains("min") ? v.at("min").to<int64_t>() : int64_t{1};
        auto sym_max = v.contains("max") ? v.at("max").to<int64_t>() : int64_t{1};
        return make_symbol(v.at("name").get_string(), sym_min, sym_max);
    }
    else if(type == "add")
    {
        auto constant = v.at("constant").to<int64_t>();
        term_map terms;
        for(const auto& t : v.at("terms"))
        {
            auto term   = node_from_value(t.at("expr"));
            auto coeff  = t.at("coeff").to<int64_t>();
            terms[term] = coeff;
        }
        return build_add(constant, std::move(terms));
    }
    else if(type == "mul")
    {
        auto coefficient = v.at("coeff").to<int64_t>();
        factor_map factors;
        for(const auto& f : v.at("factors"))
        {
            auto base     = node_from_value(f.at("expr"));
            auto exp      = f.at("exp").to<int64_t>();
            factors[base] = exp;
        }
        return build_mul(coefficient, std::move(factors));
    }
    else if(type == "tdiv")
    {
        auto num = node_from_value(v.at("num"));
        auto den = node_from_value(v.at("den"));
        return make_trunc_div(num, den);
    }
    MIGRAPHX_THROW("Unknown sym::expr node type: " + type);
}

value expr::to_value() const
{
    if(empty())
        return {};
    return node_to_value(p->node);
}

void expr::from_value(const value& v)
{
    if(v.is_null())
    {
        *this = expr{};
        return;
    }
    *this = expr{std::make_shared<impl>(node_from_value(v))};
}

} // namespace sym

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
