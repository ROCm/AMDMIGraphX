#ifndef MIGRAPHX_GUARD_RTGLIB_FILESYSTEM_HPP
#define MIGRAPHX_GUARD_RTGLIB_FILESYSTEM_HPP

#include <migraphx/config.hpp>

#if defined(__has_include) && !defined(CPPCHECK)
#if __has_include(<filesystem>) && __cplusplus >= 201703L
#define MIGRAPHX_HAS_FILESYSTEM 1
#else
#define MIGRAPHX_HAS_FILESYSTEM 0
#endif
#if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
#define MIGRAPHX_HAS_FILESYSTEM_TS 1
#else
#define MIGRAPHX_HAS_FILESYSTEM_TS 0
#endif
#else
#define MIGRAPHX_HAS_FILESYSTEM 0
#define MIGRAPHX_HAS_FILESYSTEM_TS 0
#endif

#if MIGRAPHX_HAS_FILESYSTEM
#include <filesystem>
#elif MIGRAPHX_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#else
#error "No filesystem include available"
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#if MIGRAPHX_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif MIGRAPHX_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
