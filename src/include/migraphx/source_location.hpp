#ifndef MIGRAPHX_GUARD_MIGRAPHX_SOURCE_LOCATION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SOURCE_LOCATION_HPP

#include <migraphx/config.hpp>

#if defined(CPPCHECK)
#define MIGRAPHX_HAS_SOURCE_LOCATION 1
#define MIGRAPHX_HAS_SOURCE_LOCATION_TS 1
#elif defined(__has_include)
#if __has_include(<source_location>) && __cplusplus >= 202003L
#define MIGRAPHX_HAS_SOURCE_LOCATION 1
#else
#define MIGRAPHX_HAS_SOURCE_LOCATION 0
#endif
#if __has_include(<experimental/source_location>) && __cplusplus >= 201103L
#define MIGRAPHX_HAS_SOURCE_LOCATION_TS 1
#else
#define MIGRAPHX_HAS_SOURCE_LOCATION_TS 0
#endif
#else
#define MIGRAPHX_HAS_SOURCE_LOCATION 0
#define MIGRAPHX_HAS_SOURCE_LOCATION_TS 0
#endif

#if MIGRAPHX_HAS_SOURCE_LOCATION
#include <source_location>
#elif MIGRAPHX_HAS_SOURCE_LOCATION_TS
#include <experimental/source_location>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
#if MIGRAPHX_HAS_SOURCE_LOCATION
using source_location = std::source_location;
#elif MIGRAPHX_HAS_SOURCE_LOCATION_TS
using source_location = std::experimental::source_location;
#else
struct source_location
{
    static constexpr source_location current() noexcept { return source_location{}; }
    constexpr std::uint_least32_t line() const noexcept { return 0; }
    constexpr std::uint_least32_t column() const noexcept { return 0; }
    constexpr const char* file_name() const noexcept { return ""; }
    constexpr const char* function_name() const noexcept { return ""; }
};
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SOURCE_LOCATION_HPP
