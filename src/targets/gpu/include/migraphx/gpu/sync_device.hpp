#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_SYNC_DEVICE_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_SYNC_DEVICE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct sync_device
{
    std::string name() const { return "sync_device"; }
    void apply(module& p) const;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
