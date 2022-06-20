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
    kernel(const char* image, const std::string& name);
    template <class T, MIGRAPHX_REQUIRES(sizeof(T) == 1)>
    kernel(const std::vector<T>& image, const std::string& name)
        : kernel(reinterpret_cast<const char*>(image.data()), name)
    {
    }

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                const std::vector<kernel_argument>& args) const;

    void launch(hipStream_t stream,
                std::size_t global,
                std::size_t local,
                std::vector<void*> args) const;

    auto launch(hipStream_t stream, std::size_t global, std::size_t local) const
    {
        return [=](auto&&... xs) {
            launch(stream, global, local, std::vector<kernel_argument>{xs...});
        };
    }

    private:
    std::shared_ptr<kernel_impl> impl;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
