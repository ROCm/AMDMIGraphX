#include <migraphx/gpu/kernel.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <cassert>

// extern declare the function since hip/hip_ext.h header is broken
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t, // NOLINT
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

hip_module_ptr load_module(const char* image)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, image);
    hip_module_ptr m{raw_m};
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to load module: " + hip_error(status));
    return m;
}

kernel::kernel(const char* image, const std::string& name) : impl(std::make_shared<kernel_impl>())
{
    impl->module = load_module(image);
    auto status  = hipModuleGetFunction(&impl->fun, impl->module.get(), name.c_str());
    if(hipSuccess != status)
        MIGRAPHX_THROW("Failed to get function: " + name + ": " + hip_error(status));
}

void launch_kernel(hipFunction_t fun,
                   hipStream_t stream,
                   std::size_t global,
                   std::size_t local,
                   void* kernargs,
                   std::size_t size)
{
    assert(global > 0);
    assert(local > 0);
    void* config[] = {
// HIP_LAUNCH_PARAM_* are macros that do horrible things
#ifdef MIGRAPHX_USE_CLANG_TIDY
        nullptr, kernargs, nullptr, &size, nullptr
#else
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernargs,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size,
        HIP_LAUNCH_PARAM_END
#endif
    };

    auto status = hipExtModuleLaunchKernel(
        fun, global, 1, 1, local, 1, 1, 0, stream, nullptr, reinterpret_cast<void**>(&config));
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    std::vector<void*> args) const
{
    assert(impl != nullptr);
    void* kernargs   = args.data();
    std::size_t size = args.size() * sizeof(void*);

    launch_kernel(impl->fun, stream, global, local, kernargs, size);
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    const std::vector<kernel_argument>& args) const
{
    assert(impl != nullptr);
    std::vector<char> kernargs = pack_args(args);
    std::size_t size           = kernargs.size();

    launch_kernel(impl->fun, stream, global, local, kernargs.data(), size);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
