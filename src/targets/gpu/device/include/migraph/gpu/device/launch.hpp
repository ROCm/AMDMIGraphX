#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_LAUNCH_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_LAUNCH_HPP

#include <hip/hip_runtime.h>

namespace migraph {
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

inline auto launch(std::size_t global, std::size_t local)
{
    return [=](auto f) {
        assert(local > 0);
        assert(global > 0);
        using f_type = decltype(f);
        dim3 nblocks(global / local);
        dim3 nthreads(local);
        hipLaunchKernelGGL((launcher<f_type>), nblocks, nthreads, 0, nullptr, f);
    };
}

inline auto gs_launch(std::size_t n, std::size_t local = 512)
{
    std::size_t groups  = 1 + n / local;
    std::size_t nglobal = std::min<std::size_t>(512, groups) * local;

    return [=](auto f) {
        launch(nglobal, local)([=](auto idx) {
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                f(i);
            }
        });
    };
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
