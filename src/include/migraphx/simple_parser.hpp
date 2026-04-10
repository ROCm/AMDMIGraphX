#ifndef MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP

#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/errors.hpp>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace parser {

template <class Iterator, bool AutoSkipWhitespace = true, class View = iterator_range<Iterator>>
struct simple_parser
{
    View buffer;
    Iterator pos = buffer.begin();

    static View make_view(Iterator begin, Iterator end)
    {
        if constexpr(std::is_constructible<View, decltype(std::addressof(*begin)), std::size_t>{})
        {
            auto n = std::distance(begin, end);
            if(n == 0)
                return {};
            return {std::addressof(*begin), static_cast<std::size_t>(n)};
        }
        else
        {
            return {begin, end};
        }
    }

    View peek() const
    {
        if(pos >= buffer.end())
            return {};
        return make_view(pos, buffer.end());
    }

    void advance(std::size_t n)
    {
        pos += n;
        if(pos > buffer.end())
            MIGRAPHX_THROW("Parser advanced past end of buffer");
        if constexpr(AutoSkipWhitespace)
        {
            pos = std::find_if(pos, buffer.end(), [](auto c) { return !std::isspace(c); });
        }
    }

    template <class Pred>
    View parse_while(Pred p)
    {
        auto start = pos;
        auto it    = std::find_if(pos, buffer.end(), [&](auto c) { return !p(c); });
        auto n     = std::distance(pos, it);
        advance(n);
        return make_view(start, it);
    }

    bool starts_with(const View& prefix) const
    {
        auto tail = peek();
        if(prefix.size() > tail.size())
            return false;
        else
            return std::equal(prefix.begin(), prefix.end(), tail.begin());
    }

    bool done() const { return pos >= buffer.end(); }

    bool match(const View& prefix)
    {
        if(not starts_with(prefix))
            return false;
        advance(prefix.size());
        return true;
    }

    void expect(const View& str)
    {
        if(not starts_with(str))
            MIGRAPHX_THROW(error_message("'" + std::string{str} + "'"));
        advance(str.size());
    }

    char peek_char() const
    {
        if(done())
            return '\0';
        return *pos;
    }

    std::string error_message(std::string_view expected) const
    {
        auto offset = std::distance(buffer.begin(), pos);
        return "Expected " + std::string(expected) + " at position " + std::to_string(offset) +
               " in '" + std::string(buffer) + "'";
    }
};

using simple_string_view_skip_parser =
    simple_parser<std::string_view::const_iterator, true, std::string_view>;

// ---- Parser combinator framework ----

struct skip_attr
{
};

namespace detail {

template <class T>
struct is_skip : std::is_same<T, skip_attr>
{
};

template <class T>
struct is_tuple : std::false_type
{
};
template <class... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type
{
};

template <class A, class B>
auto make_seq_result(A a, B b)
{
    if constexpr(is_skip<A>{} and is_skip<B>{})
        return skip_attr{};
    else if constexpr(is_skip<A>{})
        return std::move(b);
    else if constexpr(is_skip<B>{})
        return std::move(a);
    else if constexpr(is_tuple<A>{})
        return std::tuple_cat(std::move(a), std::make_tuple(std::move(b)));
    else
        return std::make_tuple(std::move(a), std::move(b));
}

} // namespace detail

template <class F>
struct pcomb;

template <class F>
pcomb<std::decay_t<F>> make_pcomb(F&& f)
{
    return {std::forward<F>(f)};
}

template <class F>
struct pcomb
{
    F fn;

    template <class Parser>
    auto operator()(Parser& p) const
    {
        return fn(p);
    }

    template <class Action>
    auto operator[](Action a) const
    {
        auto self = *this;
        return make_pcomb([self = std::move(self), a = std::move(a)](auto& parser) {
            auto r  = self(parser);
            using R = decltype(a(std::move(*r)));
            if(not r)
                return std::optional<R>{std::nullopt};
            return std::optional<R>{a(std::move(*r))};
        });
    }
};

template <class F>
pcomb(F) -> pcomb<F>;

// >> : sequence
template <class F1, class F2>
auto operator>>(pcomb<F1> p1, pcomb<F2> p2)
{
    return make_pcomb([p1 = std::move(p1), p2 = std::move(p2)](auto& parser) {
        using P = std::decay_t<decltype(parser)>;
        using A1 =
            typename decltype(std::declval<const pcomb<F1>&>()(std::declval<P&>()))::value_type;
        using A2 =
            typename decltype(std::declval<const pcomb<F2>&>()(std::declval<P&>()))::value_type;
        using R = decltype(detail::make_seq_result(std::declval<A1>(), std::declval<A2>()));

        auto copy = parser;
        auto r1   = p1(parser);
        if(not r1)
            return std::optional<R>{std::nullopt};
        auto r2 = p2(parser);
        if(not r2)
        {
            parser = copy;
            return std::optional<R>{std::nullopt};
        }
        return std::optional<R>{detail::make_seq_result(std::move(*r1), std::move(*r2))};
    });
}

// | : alternative
template <class F1, class F2>
auto operator|(pcomb<F1> p1, pcomb<F2> p2)
{
    return make_pcomb([p1 = std::move(p1), p2 = std::move(p2)](auto& parser) {
        using P = std::decay_t<decltype(parser)>;
        using A1 =
            typename decltype(std::declval<const pcomb<F1>&>()(std::declval<P&>()))::value_type;
        using A2 =
            typename decltype(std::declval<const pcomb<F2>&>()(std::declval<P&>()))::value_type;
        using R = std::conditional_t<std::is_same_v<A1, A2>, A1, std::variant<A1, A2>>;

        auto r1 = p1(parser);
        if(r1)
        {
            if constexpr(std::is_same_v<A1, A2>)
                return std::optional<R>{std::move(*r1)};
            else
                return std::optional<R>{R{std::in_place_index<0>, std::move(*r1)}};
        }
        auto r2 = p2(parser);
        if(r2)
        {
            if constexpr(std::is_same_v<A1, A2>)
                return std::optional<R>{std::move(*r2)};
            else
                return std::optional<R>{R{std::in_place_index<1>, std::move(*r2)}};
        }
        return std::optional<R>{std::nullopt};
    });
}

// * : zero-or-more repetition
template <class F>
auto operator*(pcomb<F> p)
{
    return make_pcomb([p = std::move(p)](auto& parser) {
        using P = std::decay_t<decltype(parser)>;
        using A =
            typename decltype(std::declval<const pcomb<F>&>()(std::declval<P&>()))::value_type;
        std::vector<A> results;
        while(true)
        {
            auto r = p(parser);
            if(not r)
                break;
            results.push_back(std::move(*r));
        }
        return std::optional{std::move(results)};
    });
}

// - : optional (always matches)
template <class F>
auto operator-(pcomb<F> p)
{
    return make_pcomb([p = std::move(p)](auto& parser) {
        using P = std::decay_t<decltype(parser)>;
        using A =
            typename decltype(std::declval<const pcomb<F>&>()(std::declval<P&>()))::value_type;
        auto r = p(parser);
        return std::optional<std::optional<A>>{r ? std::optional<A>{std::move(*r)}
                                                 : std::optional<A>{std::nullopt}};
    });
}

// ---- Combinator factories ----

inline auto lit(std::string_view s)
{
    return make_pcomb([s](auto& p) -> std::optional<skip_attr> {
        if(p.match(s))
            return skip_attr{};
        return std::nullopt;
    });
}

inline auto token(std::string_view s)
{
    return make_pcomb([s](auto& p) -> std::optional<std::string_view> {
        if(p.match(s))
            return s;
        return std::nullopt;
    });
}

template <class Pred>
auto parse_while(Pred pred)
{
    return make_pcomb([pred](auto& p) -> std::optional<std::string_view> {
        auto copy   = p;
        auto result = p.parse_while(pred);
        if(result.empty())
        {
            p = copy;
            return std::nullopt;
        }
        return result;
    });
}

template <class Pred>
auto guard(Pred pred)
{
    return make_pcomb([pred](auto& p) -> std::optional<skip_attr> {
        if(p.done() or not pred(p.peek_char()))
            return std::nullopt;
        return skip_attr{};
    });
}

template <class F>
auto lazy(F f)
{
    return pcomb{std::move(f)};
}

template <class P, class S>
auto separated_by(P p, S sep)
{
    auto cons = [](auto t) {
        auto [first, rest] = std::move(t);
        rest.insert(rest.begin(), std::move(first));
        return rest;
    };
    return (p >> *(sep >> p))[cons];
}

} // namespace parser
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP
