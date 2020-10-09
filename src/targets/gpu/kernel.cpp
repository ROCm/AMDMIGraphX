#include <migraphx/gpu/kernel.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/requires.hpp>

// extern declare the function since hip/hip_ext.h header is broken
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           size_t,
                                           hipStream_t,
                                           void**,
                                           void**,
                                           hipEvent_t = nullptr,
                                           hipEvent_t = nullptr,
                                           uint32_t   = 0);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern std::string hip_error(int error);

using hip_module_ptr = MIGRAPHX_MANAGE_PTR(hipModule_t, hipModuleUnload);

struct kernel_impl
{
    hip_module_ptr module = nullptr;
    hipFunction_t fun     = nullptr;
};

hip_module_ptr load_module(const std::vector<char>& image)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, image.data());
    hip_module_ptr m{raw_m};
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to load module: " + hip_error(status));
    return m;
}

kernel::kernel(const std::vector<char>& image, const std::string& name)
    : impl(std::make_shared<kernel_impl>())
{
    impl->module = load_module(image);
    auto status  = hipModuleGetFunction(&impl->fun, impl->module.get(), name.c_str());
    if(hipSuccess != status)
        MIGRAPHX_THROW("Failed to get function: " + name + ": " + hip_error(status));
}

template <typename T, MIGRAPHX_REQUIRES(std::is_integral<T>{})>
inline T round_up_to_next_multiple_nonnegative(T x, T y)
{
    T tmp = x + y - 1;
    return tmp - tmp % y;
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    const std::vector<std::pair<std::size_t, void*>>& args)
{
    std::vector<char> kernargs;
    for(auto&& arg : args)
    {
        std::size_t n = arg.first;
        const auto* p = static_cast<const char*>(arg.second);
        // Insert padding
        // std::size_t alignment    = arg.first;
        // std::size_t padding      = (alignment - (prev % alignment)) % alignment;
        kernargs.insert(kernargs.end(),
                        round_up_to_next_multiple_nonnegative(kernargs.size(), n) - kernargs.size(),
                        0);
        kernargs.insert(kernargs.end(), p, p + n);
    }

    std::size_t size = kernargs.size();

    void* config[] = {
// HIP_LAUNCH_PARAM_* are macros that do horrible things
#ifdef MIGRAPHX_USE_CLANG_TIDY
        nullptr, kernargs.data(), nullptr, &size, nullptr
#else
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernargs.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size,
        HIP_LAUNCH_PARAM_END
#endif
    };

    auto status = hipExtModuleLaunchKernel(impl->fun,
                                           global,
                                           1,
                                           1,
                                           local,
                                           1,
                                           1,
                                           0,
                                           stream,
                                           nullptr,
                                           reinterpret_cast<void**>(&config));
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
