#ifndef MIGRAPH_GUARD_FALLTHROUGH_HPP
#define MIGRAPH_GUARD_FALLTHROUGH_HPP

namespace migraph {

#ifdef __clang__
#define MIGRAPH_FALLTHROUGH [[clang::fallthrough]]
#else
#define MIGRAPH_FALLTHROUGH
#endif

} // namespace migraph

#endif
