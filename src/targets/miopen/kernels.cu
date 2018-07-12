
#include <hip/hip_runtime.h>
#include <migraph/operators.hpp>

namespace migraph {
namespace miopen {

template <int NDIM>
struct HIPTensorDescriptor
{
    size_t lens[NDIM];
    size_t strides[NDIM];
};

template <int NDIM>
__host__ __device__ void multiindex(size_t (&strides)[NDIM], size_t idx, size_t* result)
{
    size_t tidx = idx;
    for(size_t is = 0; is < NDIM; is++)
    {
        result[is] = tidx / strides[is];
        tidx       = tidx % strides[is];
    }
}

template <typename T, int NDIM>
__global__ void contiguous_gpu(const T* A,
                               HIPTensorDescriptor<NDIM> td_a,
                               T* At,
                               HIPTensorDescriptor<NDIM> td_at,
                               size_t nelements)
{
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements;
        i += blockDim.x * gridDim.x)
    {
        size_t s[NDIM];
        multiindex<NDIM>(td_at.strides, i, s);
        size_t lidx = 0;
        for(size_t j = 0; j < NDIM; j++)
            lidx += s[j] * td_a.strides[j];
        At[i] = A[lidx];
    }
}

migraph::argument hip_contiguous(migraph::argument arg, migraph::shape output_shape)
{
    migraph::argument result{output_shape};
    size_t ndim = output_shape.lens().size();
    visit_all(result, arg)([&](auto output, auto input) {
        if(ndim == 4)
        {
            HIPTensorDescriptor<4> td_a, td_at;
            auto s = arg.get_shape();
            for(int i = 0; i < output_shape.lens().size(); i++)
            {
                td_a.strides[i]  = s.strides().at(i);
                td_at.strides[i] = output_shape.strides().at(i);
            }
            dim3 nblocks(512);
            dim3 nthreads(512);
            hipLaunchKernelGGL((contiguous_gpu<int, 4>),
                               nblocks,
                               nthreads,
                               0,
                               0,
                               input.data(),
                               td_a,
                               output.data(),
                               td_at,
                               s.elements());
        }
        else
        {
            MIGRAPH_THROW("contiguous is only valid for 4D tensors");
        }
    });
    return result;
}
} // namespace miopen
} // namespace migraph
