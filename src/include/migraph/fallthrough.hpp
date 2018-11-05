#ifndef MIGRAPH_GUARD_FALLTHROUGH_HPP
#define MIGRAPH_GUARD_FALLTHROUGH_HPP

#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

#ifdef __clang__
#define MIGRAPH_FALLTHROUGH [[clang::fallthrough]]
#else
#define MIGRAPH_FALLTHROUGH
#endif

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
