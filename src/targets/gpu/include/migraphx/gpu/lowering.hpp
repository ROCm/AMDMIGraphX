#ifndef MIGRAPHX_GUARD_RTGLIB_MIOPEN_LOWERING_HPP
#define MIGRAPHX_GUARD_RTGLIB_MIOPEN_LOWERING_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct lowering
{
    context * ctx;
    bool offload_copy;
    std::string name() const { return "gpu::lowering"; }
    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
