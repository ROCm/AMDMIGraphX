#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP

#include <hip/hip_runtime.h>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

struct index
{
    std::size_t global = 0;
    std::size_t local = 0;
    std::size_t group = 0;

    __device__ std::size_t nglobal() const { return blockDim.x * gridDim.x; } // NOLINT

    __device__ std::size_t nlocal() const { return blockDim.x; } // NOLINT

    template <class F>
    __device__ void global_stride(std::size_t n, F f) const
    {
        const auto stride = nglobal();
        for(std::size_t i = global; i < n; i += stride)
        {
            f(i);
        }
    }

    template <class F>
    __device__ void local_stride(std::size_t n, F f) const
    {
        const auto stride = nlocal();
        for(std::size_t i = local; i < n; i += stride)
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

template <class F>
__host__ __device__ auto gs_invoke(F&& f, std::size_t i, index idx) -> decltype(f(i, idx))
{
    return f(i, idx);
}

template <class F>
__host__ __device__ auto gs_invoke(F&& f, std::size_t i, index) -> decltype(f(i))
{
    return f(i);
}

inline auto gs_launch(hipStream_t stream, std::size_t n, std::size_t local = 1024)
{
    std::size_t groups  = 1 + n / local;
    std::size_t nglobal = std::min<std::size_t>(256, groups) * local;

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
