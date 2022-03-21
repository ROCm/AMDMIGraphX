#ifndef MIGRAPHX_GUARD_KERNELS_INDEX_HPP
#define MIGRAPHX_GUARD_KERNELS_INDEX_HPP

#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

struct index
{
    index_int global = 0;
    index_int local  = 0;
    index_int group  = 0;


#ifdef MIGRAPHX_NGLOBAL
    constexpr index_constant<MIGRAPHX_NGLOBAL> nglobal() const
    {
        return {};
    }
#else
    __device__ index_int nglobal() const
    {
        return blockDim.x * gridDim.x; // NOLINT
    }
#endif

#ifdef MIGRAPHX_NLOCAL
    constexpr index_constant<MIGRAPHX_NLOCAL> nlocal() const
    {
        return {};
    }
#else
    __device__ index_int nlocal() const
    {
        return blockDim.x;             // NOLINT
    }
#endif

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

inline __device__ index make_index()
{
    return index{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x}; // NOLINT
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INDEX_HPP
