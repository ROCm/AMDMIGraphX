#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include "json_tokenize.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using token = std::pair<const char*, const char*>;
using lexer = std::function<const char*(const char* start, const char* end)>;

template<class P>
auto lex_while(P p)
{
    return [=](const char* start, const char* end) {
        return std::find_if(start, end, [&](char c) {
            return not p(c);
        });
    };
}

template<class P>
auto lex_while1(P p)
{
    return [=](const char* start, const char* end) {
        return std::find_if(start, end, [&](char c) {
            return p(c);
        });
    };
}

template<class P>
auto lex_if(P p)
{
    return [=](const char* start, const char*) {
        if (p(*start))
            return start+1;
        return start;
    };
}

std::vector<token> tokenize(const char* start, const char* end, std::vector<lexer> lexers)
{
    std::vector<token> result;
    while(start != end) 
    {
        bool error = true;
        for(auto l:lexers) 
        {
            auto next = l(start, end);
            if (next != start)
            {
                if (not std::all_of(start, next, &isspace))
                    result.emplace_back(start, next);
                start = next;
                error = false;
                break;
            }
        }
        if (error)
        {
            std::abort();
        }
    }
    return result;
}

std::vector<token> json_tokenize(const std::string& s)
{
    std::vector<lexer> lexers;

    // Quote
    lexers.push_back([](const char* start, const char* end) {
        if (*start != '\"')
            return start;
        ++start;
        while(start != end and *start != '\"')
        {
            if (*start == '\\')
                start++;
            start++;
        }

        return ++start;
    });

    lexers.push_back(lex_while(&isspace));

    // Punctation
    lexers.push_back(lex_if(&ispunct));
    
    // Identifier/number
    lexers.push_back(lex_while([](char c) { return (isalnum(c) or contains({'_', '.', '+'}, c)); }));

    return tokenize(s.data(), s.data()+s.length(), lexers);
}

std::string convert_to_json(const std::string& str)
{
    auto tokens = json_tokenize(str);
    std::stringstream ss;

    for (auto& token : tokens)
    {
        std::string s(token.first, token.second);
        if (std::isalpha(s.front()) and not contains({"null", "nan"}, s))
        {
            ss << "\"" << s << "\"";
        }
        else
        {
            ss << s;
        }
    }

    return ss.str();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
