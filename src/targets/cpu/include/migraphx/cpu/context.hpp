#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct context
{
    void finish() const {}
};

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
