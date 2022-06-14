#ifndef MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_compile_options
{
    std::size_t global;
    std::size_t local;
    std::vector<shape> inputs;
    shape output;
    std::string kernel_name           = "kernel";
    std::string params                = "";
    std::vector<shape> virtual_inputs = {};

    /**
     * @brief Set the launch parameters but allow v to override the values
     *
     * @param v A value class which can have a "global" and/or "local" keys to override the default
     * global and local
     * @param compute_global A function used to compute the global based on the local
     * @param default_local The defaul local to use if its missing from the v parameter
     */
    void set_launch_params(const value& v,
                           const std::function<std::size_t(std::size_t local)>& compute_global,
                           std::size_t default_local = 1024);

    void
    set_launch_params(const value& v, std::size_t default_global, std::size_t default_local = 1024)
    {
        set_launch_params(
            v, [=](auto) { return default_global; }, default_local);
    }
};

/// Compute global for n elements, but max out on target-specific upper limit
std::function<std::size_t(std::size_t local)>
compute_global_for(context& ctx, std::size_t n, std::size_t over = 1);

operation compile_hip_code_object(const std::string& content, hip_compile_options options);

std::size_t compute_block_size(std::size_t n, std::size_t max_block_size = 1024);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_HIP_CODE_OBJECT_HPP
