#ifndef MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_compile_options
{
    std::size_t global;
    std::size_t local;
    std::vector<shape> inputs;
    shape output;
    std::string kernel_name           = "kernel";
    std::string params                = "";
    std::vector<shape> virtual_inputs = {};
};

operation compile_hip_code_object(const std::string& content, hip_compile_options options);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP
