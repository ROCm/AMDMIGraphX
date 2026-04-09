#ifndef MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP

#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <type_traits>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace parser {

template<class Iterator, bool AutoSkipWhitespace = true, class View = iterator_range<Iterator>>
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
            pos = std::find_if(
                pos, buffer.end(), [](auto c) { return !std::isspace(c); });
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
        return "Expected " + std::string(expected) + " at position " +
               std::to_string(offset) + " in '" + std::string(buffer) + "'";
    }

    template <class F>
    bool try_parse(F f)
    {
        auto copy = *this;
        f(*this);
        if(copy.pos != pos)
            return true;
        *this = copy;
        return false;
    }

    template <class F>
    auto parse_first_of(F f) -> decltype(f(*this))
    {
        return f(*this);
    }

    template <class F, class G, class... Fs>
    auto parse_first_of(F f, G g, Fs... fs) -> decltype(f(*this))
    {
        auto copy   = *this;
        auto result = f(*this);
        if(copy.pos != pos)
            return result;
        *this = copy;
        return parse_first_of(g, fs...);
    }

    template <class F>
    auto parse_repeat(F f)
    {
        using result_type = decltype(f(*this));
        std::vector<result_type> results;
        for(;;)
        {
            auto copy = *this;
            results.push_back(f(*this));
            if(copy.pos == pos)
            {
                results.pop_back();
                break;
            }
        }
        return results;
    }
};

template <class F>
struct parser_action
{
    F fn;

    template <class Parser>
    auto operator()(Parser& p) const -> decltype(fn(p))
    {
        return fn(p);
    }
};

template <class F>
parser_action<std::decay_t<F>> action(F&& f)
{
    return {std::forward<F>(f)};
}

template <class F1, class F2>
auto operator|(parser_action<F1> a, parser_action<F2> b)
{
    return action([a = std::move(a), b = std::move(b)](auto& p) -> decltype(a(p)) {
        return p.parse_first_of(a, b);
    });
}

template <class F>
auto operator*(parser_action<F> a)
{
    return action([a = std::move(a)](auto& p) { return p.parse_repeat(a); });
}

using simple_string_view_skip_parser = simple_parser<std::string_view::const_iterator, true, std::string_view>;

} // namespace parser
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SIMPLE_PARSER_HPP
