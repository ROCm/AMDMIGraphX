
#include <hip/hip_runtime.h>
#include <migraph/operators.hpp>

namespace migraph {
namespace miopen {

template <int NDIM>
struct hip_tensor_descriptor
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
__global__ void contiguous_gpu(const T* a,
                               hip_tensor_descriptor<NDIM> a_desc,
                               T* at,
                               hip_tensor_descriptor<NDIM> at_desc,
                               size_t nelements)
{
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements;
        i += blockDim.x * gridDim.x)
    {
        size_t s[NDIM];
        multiindex<NDIM>(at_desc.strides, i, s);
        size_t lidx = 0;
        for(size_t j = 0; j < NDIM; j++)
            lidx += s[j] * a_desc.strides[j];
        at[i] = a[lidx];
    }
}

void hip_contiguous(migraph::shape output_shape, migraph::argument arg, migraph::argument result)
{
    size_t ndim = output_shape.lens().size();
    visit_all(result, arg)([&](auto output, auto input) {
        if(ndim == 4)
        {
            hip_tensor_descriptor<4> a_desc{};
            hip_tensor_descriptor<4> at_desc{};
            const auto& s = arg.get_shape();
            for(int i = 0; i < ndim; i++)
            {
                a_desc.strides[i]  = s.strides().at(i);
                at_desc.strides[i] = output_shape.strides().at(i);
            }
            dim3 nblocks(512);
            dim3 nthreads(512);
            hipLaunchKernelGGL((contiguous_gpu<int, 4>),
                               nblocks,
                               nthreads,
                               0,
                               nullptr,
                               input.data(),
                               a_desc,
                               output.data(),
                               at_desc,
                               s.elements());
        }
        else
        {
            MIGRAPH_THROW("contiguous is only valid for 4D tensors");
        }
    });
}
} // namespace miopen
} // namespace migraph
