#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP

#include <hip/hip_runtime.h>
#include <migraphx/config.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

struct index
{
    index_int global = 0;
    index_int local  = 0;
    index_int group  = 0;

    __device__ index_int nglobal() const { return blockDim.x * gridDim.x; } // NOLINT

    __device__ index_int nlocal() const { return blockDim.x; } // NOLINT

    template <class F>
    __device__ void global_stride(index_int n, F f) const
    {
        const auto stride = nglobal();
        for(index_int i = global; i < n; i += stride)
        {
            f(i);
        }
    }

    template <class F>
    __device__ void local_stride(index_int n, F f) const
    {
        const auto stride = nlocal();
        for(index_int i = local; i < n; i += stride)
        {
            f(i);
        }
    }
};

template <class F>
__global__ void launcher(F f)
{
    index idx{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x};
    f(idx);
}

inline auto launch(hipStream_t stream, index_int global, index_int local)
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

template <class F>
MIGRAPHX_DEVICE_CONSTEXPR auto gs_invoke(F&& f, index_int i, index idx) -> decltype(f(i, idx))
{
    return f(i, idx);
}

template <class F>
MIGRAPHX_DEVICE_CONSTEXPR auto gs_invoke(F&& f, index_int i, index) -> decltype(f(i))
{
    return f(i);
}

inline auto gs_launch(hipStream_t stream, index_int n, index_int local = 1024)
{
    index_int groups  = (n + local - 1) / local;
    index_int nglobal = std::min<index_int>(256, groups) * local;

    return [=](auto f) {
        launch(stream, nglobal, local)(
            [=](auto idx) { idx.global_stride(n, [&](auto i) { gs_invoke(f, i, idx); }); });
    };
}

// Workaround hcc's broken tile_static macro
#ifdef tile_static
#undef tile_static
#define MIGRAPHX_DEVICE_SHARED __attribute__((tile_static))
#else
#define MIGRAPHX_DEVICE_SHARED __shared__
#endif

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
