#ifndef MIGRAPHX_GUARD_FALLTHROUGH_HPP
#define MIGRAPHX_GUARD_FALLTHROUGH_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef __clang__
#define MIGRAPHX_FALLTHROUGH [[clang::fallthrough]]
#else
#define MIGRAPHX_FALLTHROUGH
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
