#ifndef MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP
#define MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct prepare_reduce
{
    std::string name() const { return "gpu::prepare_reduce"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP


