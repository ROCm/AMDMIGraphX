#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_LAUNCH_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_LAUNCH_HPP

#include <hip/hip_runtime.h>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

struct index
{
    std::size_t global;
    std::size_t local;
    std::size_t group;
};

template <class F>
__global__ void launcher(F f)
{
    index idx{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x};
    f(idx);
}

inline auto launch(hipStream_t stream, std::size_t global, std::size_t local)
{
    return [=](auto f) {
        assert(local > 0);
        assert(global > 0);
        using f_type = decltype(f);
        dim3 nblocks(global / local);
        dim3 nthreads(local);
        hipLaunchKernelGGL((launcher<f_type>), nblocks, nthreads, 0, stream, f);
    };
}

inline auto gs_launch(hipStream_t stream, std::size_t n, std::size_t local = 1024)
{
    std::size_t groups  = 1 + n / local;
    std::size_t nglobal = std::min<std::size_t>(256, groups) * local;

    return [=](auto f) {
        launch(stream, nglobal, local)([=](auto idx) {
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                f(i);
            }
        });
    };
}

// Workaround hcc's broken tile_static macro
#ifdef tile_static
#undef tile_static
#define MIGRAPH_DEVICE_SHARED __attribute__((tile_static))
#else
#define MIGRAPH_DEVICE_SHARED __shared__
#endif

} // namespace device
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
