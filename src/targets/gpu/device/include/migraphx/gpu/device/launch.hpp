#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_LAUNCH_HPP

#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_prof_str.h>
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

    // default value of the variable "local", equals number of threads per compute unit
    // when launching kernels.
    // preliminary testing of pointwise operators indicates multiples of 64 or 256 work best.
    // Max allowable value is 1024
    #define LOCAL_THREADS 256

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
    index idx{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x}; // NOLINT
    f(idx);
}

// global:  number of global threads (?)
// local: number of local threads  (?)
inline auto launch(hipStream_t stream, index_int global, index_int local)
{
    return [=](auto f) {
        assert(local > 0);
        assert(global > 0);
        using f_type = decltype(f);
        dim3 nblocks(global / local);
        // printf("global:   %d nblocks:  %d ++++++++++++\n",global, global / local);
        dim3 nthreads(local);
        // cppcheck-suppress UseDeviceLaunch
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

// n:  number of elements (tensor size)
// global:  number of global work items (threads)
// local: number of local work items (threads) per compute unit (CU) 
inline auto gs_launch(hipStream_t stream, index_int n, index_int local = LOCAL_THREADS)
{
    index_int groups = (n + local - 1) / local;
    // max possible number of blocks is set to 1B (1,073,741,824)
    index_int nglobal = std::min<index_int>(1073741824, groups) * local;

    return [=](auto f) {
        launch(stream, nglobal, local)([=](auto idx) __device__ {
            idx.global_stride(n, [&](auto i) { gs_invoke(f, i, idx); });
        });
    };
}

#ifdef MIGRAPHX_USE_CLANG_TIDY
#define MIGRAPHX_DEVICE_SHARED
#else
// Workaround hcc's broken tile_static macro
#ifdef tile_static
#undef tile_static
#define MIGRAPHX_DEVICE_SHARED __attribute__((tile_static))
#else
#define MIGRAPHX_DEVICE_SHARED __shared__
#endif
#endif

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
