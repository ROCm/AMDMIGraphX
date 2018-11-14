#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

struct lowering
{
    context ctx;
    std::string name() const { return "gpu::lowering"; }
    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
