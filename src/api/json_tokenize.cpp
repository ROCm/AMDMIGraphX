#include "json_tokenize.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

siter colon(siter start, siter end)
{
    return std::find_if(start, end, [](auto c) { return c == ':'; });
}

std::pair<siter, siter> key(siter start, siter end)
{
    // find key end
    --end;
    while(start != end)
    {
        if(*end != ' ')
        {
            break;
        }
        --end;
    }
    auto ke = end;

    if(start == end)
    {
        if(*ke != '\"')
        {
            return {ke, ke};
        }
        else
        {
            MIGRAPHX_THROW("KEY: single quote cannot be a key!");
        }
    }

    // find key start
    --end;
    while(start != end)
    {
        // match
        if(*ke == '\"' and *end == '\"')
        {
            break;
        }
        else if(*ke != '\"' and (std::ispunct(*end) or *end == ' '))
        {
            ++end;
            break;
        }
        --end;
    }
    auto ks = end;

    return {ks, ke};
}

std::string json_tokenize(const std::string& s)
{
    siter start = s.begin();
    siter end   = s.end();
    std::vector<token> tokens;

    while(start != end)
    {
        auto colon_iter = colon(start, end);
        if(colon_iter == s.end())
        {
            break;
        }

        if(start != colon_iter)
        {
            auto tk = key(start, colon_iter);
            if(tk.first != tk.second)
            {
                // key is not quoted yet, need to quote it
                if(*tk.first != '\"' or *tk.second != '\"')
                {
                    tokens.emplace_back(tk);
                }
            }
        }

        start = ++colon_iter;
    }

    std::string result;
    siter prev_it = s.begin();
    for(auto token : tokens)
    {
        result.append(std::string(prev_it, token.first));
        result.append(1, '"');
        result.append(std::string(token.first, ++token.second));
        result.append(1, '"');
        prev_it = token.second;
    }
    result.append(std::string(prev_it, s.end()));

    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
