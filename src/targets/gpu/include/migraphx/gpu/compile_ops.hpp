#ifndef MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/allocation_model.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct context;

struct compile_ops
{
    context* ctx = nullptr;
    gpu_allocation_model alloc{false};
    std::string name() const { return "gpu::compile_ops"; }
    void apply(module& m) const;
};

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
