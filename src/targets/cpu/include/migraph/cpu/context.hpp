#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct context
{
    void finish() const {}
};

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
