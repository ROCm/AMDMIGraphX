#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_GPU_ALLOCATION_MODEL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_GPU_ALLOCATION_MODEL_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/instruction_ref.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct gpu_allocation_model
{
    std::string name() const;
    std::string copy() const;
    operation allocate(const shape& s) const;
    operation preallocate(const shape& s, const std::string& id) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
