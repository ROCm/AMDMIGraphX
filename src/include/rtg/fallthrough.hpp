#ifndef RTG_GUARD_FALLTHROUGH_HPP
#define RTG_GUARD_FALLTHROUGH_HPP

namespace rtg {

#ifdef __clang__
#define RTG_FALLTHROUGH [[clang::fallthrough]]
#else
#define RTG_FALLTHROUGH
#endif

} // namespace rtg

#endif
