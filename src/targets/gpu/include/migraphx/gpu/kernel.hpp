#ifndef MIGRAPHX_GUARD_RTGLIB_KERNEL_HPP
#define MIGRAPHX_GUARD_RTGLIB_KERNEL_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct kernel_impl;

struct kernel
{
    kernel() = default;
    kernel(const std::vector<char>& image, const std::string& name);

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                const std::vector<kernel_argument>& args);

    auto launch(hipStream_t stream, std::size_t global, std::size_t local)
    {
        return [=](auto&&... xs) { launch(stream, global, local, {xs...}); };
    }

    private:
    std::shared_ptr<kernel_impl> impl;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
