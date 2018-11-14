#ifndef MIGRAPH_GUARD_RTGLIB_TYPE_NAME_HPP
#define MIGRAPH_GUARD_RTGLIB_TYPE_NAME_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

template <class PrivateMigraphTypeNameProbe>
const std::string& get_type_name()
{
    static std::string name;

    if(name.empty())
    {
#ifdef _MSC_VER
        name = typeid(PrivateMigraphTypeNameProbe).name();
        name = name.substr(7);
#else
        const char parameter_name[] = "PrivateMigraphTypeNameProbe =";

        name = __PRETTY_FUNCTION__;

        auto begin  = name.find(parameter_name) + sizeof(parameter_name);
#if(defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7)
        auto length = name.find_last_of(",") - begin;
#else
        auto length = name.find_first_of("];", begin) - begin;
#endif
        name        = name.substr(begin, length);
#endif
    }

    return name;
}

template <class T>
const std::string& get_type_name(const T&)
{
    return migraphx::get_type_name<T>();
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
