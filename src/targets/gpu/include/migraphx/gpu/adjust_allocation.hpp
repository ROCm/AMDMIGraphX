#ifndef MIGRAPHX_GUARD_RTGLIB_ADJUST_ALLOCATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_ADJUST_ALLOCATION_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

struct adjust_allocation
{
    std::string name() const { return "gpu::adjust_allocation"; }
    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
