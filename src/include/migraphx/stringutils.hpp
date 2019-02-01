#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_STRINGUTILS_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_STRINGUTILS_HPP

#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline std::string
replace_string(std::string subject, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

inline bool ends_with(const std::string& value, const std::string& suffix)
{
    if(suffix.size() > value.size())
        return false;
    else
        return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

template <class Strings>
inline std::string join_strings(Strings strings, const std::string& delim)
{
    auto it = strings.begin();
    if(it == strings.end())
        return "";

    auto nit = std::next(it);
    return std::accumulate(
        nit, strings.end(), *it, [&](std::string x, std::string y) { return x + delim + y; });
}

template <class F>
inline std::string transform_string(std::string s, F f)
{
    std::transform(s.begin(), s.end(), s.begin(), f);
    return s;
}

inline std::string to_upper(std::string s) { return transform_string(std::move(s), ::toupper); }

inline bool starts_with(const std::string& value, const std::string& prefix)
{
    if(prefix.size() > value.size())
        return false;
    else
        return std::equal(prefix.begin(), prefix.end(), value.begin());
}

inline std::string remove_prefix(std::string s, const std::string& prefix)
{
    if(starts_with(s, prefix))
        return s.substr(prefix.length());
    else
        return s;
}

template <class Range>
inline std::string to_string_range(const Range& r)
{
    std::stringstream ss;
    if(!r.empty())
    {
        ss << r.front();
        std::for_each(std::next(r.begin()), r.end(), [&](auto&& x) { ss << ", " << x; });
    }
    return ss.str();
}

template <class T>
inline std::string to_string(const T& x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
