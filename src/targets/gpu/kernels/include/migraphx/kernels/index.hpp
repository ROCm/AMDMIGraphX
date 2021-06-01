#ifndef MIGRAPHX_GUARD_KERNELS_INDEX_HPP
#define MIGRAPHX_GUARD_KERNELS_INDEX_HPP

#include <hip/hip_runtime.h>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

struct index
{
    index_int global = 0;
    index_int local  = 0;
    index_int group  = 0;

    __device__ index_int nglobal() const { 
#ifdef MIGRAPHX_NGLOBAL
        return MIGRAPHX_NGLOBAL;
#else
        return blockDim.x * gridDim.x;
#endif
    }

    __device__ index_int nlocal() const {
#ifdef MIGRAPHX_NLOCAL
        return MIGRAPHX_NLOCAL;
#else
        return blockDim.x;
#endif
    }
};

inline __device__ index make_index()
{
    return index{blockIdx.x * blockDim.x + threadIdx.x, threadIdx.x, blockIdx.x};
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INDEX_HPP
