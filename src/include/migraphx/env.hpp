#ifndef MIGRAPHX_GUARD_RTGLIB_ENV_HPP
#define MIGRAPHX_GUARD_RTGLIB_ENV_HPP

#include <vector>
#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Declare a cached environment variable
#define MIGRAPHX_DECLARE_ENV_VAR(x)               \
    struct x                                      \
    {                                             \
        static const char* value() { return #x; } \
    }; // NOLINT

bool enabled(const char* name);
bool disabled(const char* name);
std::vector<std::string> env(const char* name);

template <class T>
bool enabled(T)
{
    static const bool result = enabled(T::value());
    return result;
}

template <class T>
bool disabled(T)
{
    static const bool result = disabled(T::value());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
