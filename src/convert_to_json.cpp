#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/convert_to_json.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using token = std::pair<const char*, const char*>;
using lexer = std::function<const char*(const char* start, const char* end)>;

template <class P>
auto lex_while(P p)
{
    return [=](const char* start, const char* end) {
        return std::find_if(start, end, [&](char c) { return not p(c); });
    };
}

template <class P>
auto lex_if(P p)
{
    return [=](const char* start, const char*) {
        if(p(*start))
            return start + 1;
        return start;
    };
}

std::vector<token> tokenize(const char* start, const char* end, const std::vector<lexer>& lexers)
{
    std::vector<token> result;
    while(start != end)
    {
        bool error = true;
        for(const auto& l : lexers)
        {
            const auto* next = l(start, end);
            if(next != start)
            {
                result.emplace_back(start, next);
                start = next;
                error = false;
                break;
            }
        }

        if(error)
        {
            MIGRAPHX_THROW("TOKENIZE: no token found!");
        }
    }

    return result;
}

std::vector<token> json_tokenize(const std::string& s)
{
    std::vector<lexer> lexers;

    // Quote
    lexers.push_back([](const char* start, const char* end) {
        if(*start != '\"')
            return start;
        ++start;
        while((start != end) and (*start != '\"'))
        {
            if(*start == '\\')
                start++;
            start++;
        }

        return ++start;
    });

    // Line comments
    lexers.push_back([](const char* start, const char* end) {
        if(*start == '#')
            start++;
        else if((start + 1) < end and start[0] == '/' and start[1] == '/')
            start += 2;
        else
            return start;
        return std::find_if(start, end, [&](char c) { return c == '\n'; });
    });

    // Whitespace
    lexers.push_back(lex_while(&isspace));

    // Punctation
    lexers.push_back(lex_if(&ispunct));

    // Identifier/number
    lexers.push_back(lex_while([](char c) {
        return (isalnum(c) != 0 or contains({'_', '.', '+'}, c));
    }));

    return tokenize(s.data(), s.data() + s.length(), lexers);
}

std::string convert_to_json(const std::string& str)
{
    auto tokens = json_tokenize(str);
    std::stringstream ss;

    for(auto& token : tokens)
    {
        std::string s(token.first, token.second);
        if(starts_with(s, "#") or starts_with(s, "//"))
            continue;
        if(std::isalpha(s.front()) != 0 and
           not contains({"null", "nan", "true", "false", "inf"}, s))
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
