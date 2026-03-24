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

#include <migraphx/symbolic.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>

#include <algorithm>
#include <cassert>
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
struct fdiv_data
{
    expr_ptr numerator;
    expr_ptr denominator;
};

using expr_data = std::variant<integer_data, symbol_data, add_data, mul_data, fdiv_data>;

constexpr int kind_integer = 0;
constexpr int kind_symbol  = 1;
constexpr int kind_add     = 2;
constexpr int kind_mul     = 3;
constexpr int kind_fdiv    = 4;

struct expr_node
{
    expr_data data;
    std::size_t cached_hash = 0;

    int kind() const { return static_cast<int>(data.index()); }
};

static bool is_integer(const expr_ptr& e) { return e->kind() == kind_integer; }
static bool is_symbol(const expr_ptr& e) { return e->kind() == kind_symbol; }
static bool is_add(const expr_ptr& e) { return e->kind() == kind_add; }
static bool is_mul(const expr_ptr& e) { return e->kind() == kind_mul; }
static bool is_fdiv(const expr_ptr& e) { return e->kind() == kind_fdiv; }

static int64_t get_integer(const expr_ptr& e) { return std::get<integer_data>(e->data).value; }
static const std::string& get_symbol_name(const expr_ptr& e)
{
    return std::get<symbol_data>(e->data).name;
}
static const add_data& get_add(const expr_ptr& e) { return std::get<add_data>(e->data); }
static const mul_data& get_mul(const expr_ptr& e) { return std::get<mul_data>(e->data); }
static const fdiv_data& get_fdiv(const expr_ptr& e) { return std::get<fdiv_data>(e->data); }

// ===================================================================
// Section 2: Hash computation
// ===================================================================

static std::size_t hash_combine(std::size_t seed, std::size_t v)
{
    return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
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
    std::size_t h = std::hash<int>{}(static_cast<int>(d.index()));
    if(auto* p = std::get_if<integer_data>(&d))
        return hash_combine(h, std::hash<int64_t>{}(p->value));
    if(auto* p = std::get_if<symbol_data>(&d))
        return hash_combine(h, std::hash<std::string>{}(p->name));
    if(auto* p = std::get_if<add_data>(&d))
    {
        h = hash_combine(h, std::hash<int64_t>{}(p->constant));
        return hash_combine(h, hash_ordered_map(p->terms));
    }
    if(auto* p = std::get_if<mul_data>(&d))
    {
        h = hash_combine(h, std::hash<int64_t>{}(p->coefficient));
        return hash_combine(h, hash_ordered_map(p->factors));
    }
    if(auto* p = std::get_if<fdiv_data>(&d))
    {
        h = hash_combine(h, p->numerator->cached_hash);
        return hash_combine(h, p->denominator->cached_hash);
    }
    return h;
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
    if(a->kind() != b->kind())
        return a->kind() < b->kind() ? -1 : 1;

    switch(a->kind())
    {
    case kind_integer: {
        auto va = get_integer(a);
        auto vb = get_integer(b);
        return (va < vb) ? -1 : (va > vb) ? 1 : 0;
    }
    case kind_symbol: {
        const auto& na = get_symbol_name(a);
        const auto& nb = get_symbol_name(b);
        return na.compare(nb);
    }
    case kind_add: {
        const auto& da = get_add(a);
        const auto& db = get_add(b);
        if(da.constant != db.constant)
            return da.constant < db.constant ? -1 : 1;
        return compare_maps(da.terms, db.terms);
    }
    case kind_mul: {
        const auto& da = get_mul(a);
        const auto& db = get_mul(b);
        if(da.coefficient != db.coefficient)
            return da.coefficient < db.coefficient ? -1 : 1;
        return compare_maps(da.factors, db.factors);
    }
    case kind_fdiv: {
        const auto& da = get_fdiv(a);
        const auto& db = get_fdiv(b);
        int c = compare_expr(da.numerator, db.numerator);
        if(c != 0)
            return c;
        return compare_expr(da.denominator, db.denominator);
    }
    default: return 0;
    }
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
    auto n     = std::make_shared<expr_node>();
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

static expr_ptr make_symbol(const std::string& name) { return make_node(symbol_data{name}); }

static expr_ptr make_add(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_sub(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_neg(const expr_ptr& a);
static expr_ptr make_mul(const expr_ptr& a, const expr_ptr& b);
static expr_ptr make_floor_div(const expr_ptr& a, const expr_ptr& b);
static expr_ptr build_mul(int64_t coefficient, factor_map factors);

struct add_parts
{
    int64_t constant = 0;
    term_map terms;
};

static add_parts extract_add(const expr_ptr& e)
{
    if(is_integer(e))
        return {get_integer(e), {}};
    if(is_add(e))
    {
        const auto& d = get_add(e);
        return {d.constant, d.terms};
    }
    if(is_mul(e))
    {
        const auto& d = get_mul(e);
        auto base = build_mul(1, d.factors);
        return {0, {{base, d.coefficient}}};
    }
    return {0, {{e, 1}}};
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
    if(is_integer(a))
        return make_integer(-get_integer(a));
    if(is_add(a))
    {
        const auto& d = get_add(a);
        term_map negated;
        for(const auto& [term, coeff] : d.terms)
            negated[term] = -coeff;
        return build_add(-d.constant, std::move(negated));
    }
    if(is_mul(a))
    {
        const auto& d = get_mul(a);
        return make_node(mul_data{-d.coefficient, d.factors});
    }
    return make_mul(make_integer(-1), a);
}

static expr_ptr make_sub(const expr_ptr& a, const expr_ptr& b)
{
    return make_add(a, make_neg(b));
}

struct mul_parts
{
    int64_t coefficient = 1;
    factor_map factors;
};

static mul_parts extract_mul(const expr_ptr& e)
{
    if(is_integer(e))
        return {get_integer(e), {}};
    if(is_mul(e))
    {
        const auto& d = get_mul(e);
        return {d.coefficient, d.factors};
    }
    return {1, {{e, 1}}};
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
    // Special case: multiplying an integer by an Add → scale the Add
    if(is_integer(a) and is_add(b))
    {
        int64_t n       = get_integer(a);
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
    if(is_integer(b) and is_add(a))
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

static expr_ptr make_floor_div(const expr_ptr& a, const expr_ptr& b)
{
    if(is_integer(b))
    {
        int64_t den = get_integer(b);
        if(den == 0)
            MIGRAPHX_THROW("symbolic: division by zero");
        if(den == 1)
            return a;
        if(is_integer(a))
            return make_integer(get_integer(a) / den);
        // Cancel exact coefficient in Mul: (c*factors)/den where c%den==0
        if(is_mul(a))
        {
            const auto& d = get_mul(a);
            if(d.coefficient % den == 0)
                return build_mul(d.coefficient / den, d.factors);
        }
    }

    return make_node(fdiv_data{a, b});
}

// ===================================================================
// Section 6: Substitution and evaluation
// ===================================================================

// Partial substitution: replaces bound symbols with integers and re-canonicalizes.
// Unbound symbols are left as-is, producing a simplified symbolic expression.
static expr_ptr substitute(const expr_ptr& e,
                           const std::map<std::string, int64_t>& bindings)
{
    switch(e->kind())
    {
    case kind_integer: return e;
    case kind_symbol: {
        auto it = bindings.find(get_symbol_name(e));
        if(it != bindings.end())
            return make_integer(it->second);
        return e;
    }
    case kind_add: {
        const auto& d = get_add(e);
        expr_ptr result = make_integer(d.constant);
        for(const auto& [term, coeff] : d.terms)
        {
            auto st = substitute(term, bindings);
            result  = make_add(result, make_mul(make_integer(coeff), st));
        }
        return result;
    }
    case kind_mul: {
        const auto& d = get_mul(e);
        expr_ptr result = make_integer(d.coefficient);
        for(const auto& [base, exp] : d.factors)
        {
            auto sb = substitute(base, bindings);
            for(int64_t i = 0; i < exp; ++i)
                result = make_mul(result, sb);
        }
        return result;
    }
    case kind_fdiv: {
        const auto& d = get_fdiv(e);
        auto sn       = substitute(d.numerator, bindings);
        auto sd       = substitute(d.denominator, bindings);
        return make_floor_div(sn, sd);
    }
    default: return e;
    }
}

// Full evaluation: computes integer result directly without allocations.
// All symbols must be bound; throws if any symbol is unbound.
static int64_t eval_direct(const expr_ptr& e,
                           const std::map<std::string, std::size_t>& symbol_map)
{
    switch(e->kind())
    {
    case kind_integer: return get_integer(e);
    case kind_symbol: {
        auto it = symbol_map.find(get_symbol_name(e));
        if(it != symbol_map.end())
            return static_cast<int64_t>(it->second);
        MIGRAPHX_THROW("symbolic_expr::eval: unbound symbol '" + get_symbol_name(e) + "'");
    }
    case kind_add: {
        const auto& d = get_add(e);
        int64_t sum   = d.constant;
        for(const auto& [term, coeff] : d.terms)
            sum += coeff * eval_direct(term, symbol_map);
        return sum;
    }
    case kind_mul: {
        const auto& d = get_mul(e);
        int64_t prod  = d.coefficient;
        for(const auto& [base, exp] : d.factors)
        {
            int64_t val = eval_direct(base, symbol_map);
            for(int64_t i = 0; i < exp; ++i)
                prod *= val;
        }
        return prod;
    }
    case kind_fdiv: {
        const auto& d = get_fdiv(e);
        return eval_direct(d.numerator, symbol_map) / eval_direct(d.denominator, symbol_map);
    }
    default: return 0;
    }
}

static std::size_t evaluate(const expr_ptr& e,
                            const std::map<std::string, std::size_t>& symbol_map)
{
    return static_cast<std::size_t>(eval_direct(e, symbol_map));
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

static std::string print_expr(const expr_ptr& e, int parent_prec)
{
    switch(e->kind())
    {
    case kind_integer: return std::to_string(get_integer(e));
    case kind_symbol: return get_symbol_name(e);
    case kind_add: {
        const auto& d = get_add(e);
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
    case kind_mul: {
        const auto& d = get_mul(e);
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
            if(not first)
                os << "*";
            if(exp == 1)
                os << print_expr(base, prec_mul + 1);
            else
                os << print_expr(base, prec_mul + 1) << "**" << exp;
            first = false;
        }
        std::string raw = os.str();
        if(parent_prec > prec_mul)
            return "(" + raw + ")";
        return raw;
    }
    case kind_fdiv: {
        const auto& d = get_fdiv(e);
        std::string s = print_expr(d.numerator, prec_mul + 1) + "/" +
                        print_expr(d.denominator, prec_mul + 1);
        if(parent_prec > prec_mul)
            return "(" + s + ")";
        return s;
    }
    default: return "?";
    }
}

// ===================================================================
// Section 8: Recursive descent parser
// ===================================================================

static void skip_ws(const char*& p)
{
    while(*p and std::isspace(static_cast<unsigned char>(*p)))
        ++p;
}

static expr_ptr parse_expr(const char*& p);
static expr_ptr parse_term(const char*& p);
static expr_ptr parse_unary(const char*& p);
static expr_ptr parse_primary(const char*& p);

static expr_ptr parse_primary(const char*& p)
{
    skip_ws(p);
    if(std::isdigit(static_cast<unsigned char>(*p)))
    {
        int64_t n = 0;
        while(std::isdigit(static_cast<unsigned char>(*p)))
        {
            n = n * 10 + (*p - '0');
            ++p;
        }
        return make_integer(n);
    }
    if(std::isalpha(static_cast<unsigned char>(*p)) or *p == '_')
    {
        std::string name;
        while(std::isalnum(static_cast<unsigned char>(*p)) or *p == '_')
        {
            name += *p;
            ++p;
        }
        if(name == "floor")
        {
            skip_ws(p);
            if(*p != '(')
                MIGRAPHX_THROW("symbolic parser: expected '(' after 'floor'");
            ++p;
            auto inner = parse_expr(p);
            skip_ws(p);
            if(*p != ')')
                MIGRAPHX_THROW("symbolic parser: expected ')' after floor argument");
            ++p;
            return inner;
        }
        return make_symbol(name);
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
            left = make_floor_div(left, parse_unary(p));
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
// Section 9: symbolic_expr public API wrapper
// ===================================================================

struct symbolic_expr::impl
{
    expr_ptr node;

    impl() : node(make_integer(0)) {}
    explicit impl(expr_ptr e) : node(std::move(e)) {}
};

symbolic_expr::symbolic_expr() = default;

symbolic_expr::symbolic_expr(std::shared_ptr<const impl> pi) : p(std::move(pi)) {}

symbolic_expr::symbolic_expr(std::size_t n)
    : p(std::make_shared<impl>(make_integer(static_cast<int64_t>(n))))
{
}

symbolic_expr::symbolic_expr(const std::string& s)
{
    if(s.empty())
        return;
    p = std::make_shared<impl>(parse_string(s));
}

bool symbolic_expr::empty() const { return p == nullptr; }

std::string symbolic_expr::to_string() const
{
    if(empty())
        return {};
    return print_expr(p->node);
}

std::size_t symbolic_expr::eval(const std::map<std::string, std::size_t>& symbol_map) const
{
    if(empty())
        return 0;
    return evaluate(p->node, symbol_map);
}

symbolic_expr symbolic_expr::subs(const std::map<std::string, std::size_t>& symbol_map) const
{
    if(empty())
        return {};
    std::map<std::string, int64_t> bindings;
    for(const auto& [k, v] : symbol_map)
        bindings[k] = static_cast<int64_t>(v);
    auto result = substitute(p->node, bindings);
    return {std::make_shared<impl>(std::move(result))};
}

symbolic_expr operator+(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->node : make_integer(0);
    auto eb = b.p ? b.p->node : make_integer(0);
    return {std::make_shared<symbolic_expr::impl>(make_add(ea, eb))};
}

symbolic_expr operator-(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->node : make_integer(0);
    auto eb = b.p ? b.p->node : make_integer(0);
    return {std::make_shared<symbolic_expr::impl>(make_sub(ea, eb))};
}

symbolic_expr operator*(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->node : make_integer(0);
    auto eb = b.p ? b.p->node : make_integer(0);
    return {std::make_shared<symbolic_expr::impl>(make_mul(ea, eb))};
}

symbolic_expr operator/(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->node : make_integer(0);
    auto eb = b.p ? b.p->node : make_integer(0);
    return {std::make_shared<symbolic_expr::impl>(make_floor_div(ea, eb))};
}

bool operator==(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return true;
    if(a.empty() != b.empty())
        return false;
    return expr_equal(a.p->node, b.p->node);
}

bool operator!=(const symbolic_expr& a, const symbolic_expr& b) { return not(a == b); }

std::ostream& operator<<(std::ostream& os, const symbolic_expr& e)
{
    if(not e.empty())
        os << print_expr(e.p->node);
    return os;
}

void migraphx_to_value(value& v, const symbolic_expr& e)
{
    v = migraphx::to_value(e.to_string());
}

void migraphx_from_value(const value& v, symbolic_expr& e)
{
    auto s = v.get_string();
    if(s.empty())
        e = symbolic_expr{};
    else
        e = symbolic_expr(s);
}

// ===================================================================
// Section 10: dynamic_dimension arithmetic (unchanged from original)
// ===================================================================

using dd = shape::dynamic_dimension;

dd& dd::operator+=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) + x.sym.value_or(symbolic_expr(x.min));
    min    = min + x.min;
    max    = (max > std::numeric_limits<std::size_t>::max() - x.max)
                 ? std::numeric_limits<std::size_t>::max()
                 : max + x.max;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return o + x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator-=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) - x.sym.value_or(symbolic_expr(x.min));
    min    = (min > x.max) ? min - x.max : 0;
    max    = (max > x.min) ? max - x.min : 0;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return (o > x.min) ? o - x.min : 0; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator*=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) * x.sym.value_or(symbolic_expr(x.min));
    min    = min * x.min;
    max    = (max > std::numeric_limits<std::size_t>::max() / (x.max == 0 ? 1 : x.max))
                 ? std::numeric_limits<std::size_t>::max()
                 : max * x.max;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return o * x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator/=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) / x.sym.value_or(symbolic_expr(x.min));
    min    = (x.max == 0) ? 0 : min / x.max;
    max    = (x.min == 0) ? std::numeric_limits<std::size_t>::max() : max / x.min;
    if(x.is_fixed())
    {
        std::set<std::size_t> new_optimals;
        std::transform(optimals.begin(),
                       optimals.end(),
                       std::inserter(new_optimals, new_optimals.begin()),
                       [&](auto o) { return (x.min == 0) ? std::size_t{0} : o / x.min; });
        optimals = new_optimals;
    }
    else
    {
        optimals.clear();
    }
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd operator+(const dd& x, const dd& y)
{
    auto result = x;
    result += y;
    return result;
}

dd operator-(const dd& x, const dd& y)
{
    auto result = x;
    result -= y;
    return result;
}

dd operator*(const dd& x, const dd& y)
{
    auto result = x;
    result *= y;
    return result;
}

dd operator/(const dd& x, const dd& y)
{
    auto result = x;
    result /= y;
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
