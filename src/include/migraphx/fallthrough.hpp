#ifndef MIGRAPH_GUARD_FALLTHROUGH_HPP
#define MIGRAPH_GUARD_FALLTHROUGH_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

#ifdef __clang__
#define MIGRAPH_FALLTHROUGH [[clang::fallthrough]]
#else
#define MIGRAPH_FALLTHROUGH
#endif

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
