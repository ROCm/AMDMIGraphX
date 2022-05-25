#ifndef MIGRAPHX_GUARD_GPU_PREFUSE_OPS_HPP
#define MIGRAPHX_GUARD_GPU_PREFUSE_OPS_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct prefuse_ops
{
    std::string name() const { return "gpu::prefuse_ops"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PREFUSE_OPS_HPP
