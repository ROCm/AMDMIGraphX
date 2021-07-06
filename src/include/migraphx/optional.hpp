#ifndef MIGRAPHX_GUARD_MIGRAPHX_OPTIONAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_OPTIONAL_HPP

#include <migraphx/config.hpp>

#if defined(__has_include) && !defined(CPPCHECK)
#if __has_include(<optional>) && __cplusplus >= 201703L
#define MIGRAPHX_HAS_OPTIONAL 1
#else
#define MIGRAPHX_HAS_OPTIONAL 0
#endif
#if __has_include(<experimental/optional>) && __cplusplus >= 201103L
#define MIGRAPHX_HAS_OPTIONAL_TS 1
#else
#define MIGRAPHX_HAS_OPTIONAL_TS 0
#endif
#else
#define MIGRAPHX_HAS_OPTIONAL 0
#define MIGRAPHX_HAS_OPTIONAL_TS 0
#endif

#if MIGRAPHX_HAS_OPTIONAL
#include <optional>
#elif MIGRAPHX_HAS_OPTIONAL_TS
#include <experimental/optional>
#else
#error "No optional include available"
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#if MIGRAPHX_HAS_OPTIONAL
template <class T>
using optional         = std::optional<T>;
using nullopt_t        = std::nullopt_t;
constexpr auto nullopt = std::nullopt;
#elif MIGRAPHX_HAS_OPTIONAL_TS
template <class T>
using optional         = std::experimental::optional<T>;
using nullopt_t        = std::experimental::nullopt_t;
constexpr auto nullopt = std::experimental::nullopt;
#endif

template <class T>
bool has_value(const optional<T>& x)
{
    return x != nullopt;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_OPTIONAL_HPP
