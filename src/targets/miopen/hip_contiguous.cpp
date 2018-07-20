
#include <hip/hip_runtime.h>
#include <migraph/operators.hpp>

namespace migraph {
namespace miopen {

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 0:
    {
        f(std::integral_constant<std::size_t, 0>{});
        break;
    }
    case 1:
    {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

template <size_t NDim>
struct hip_index
{
    size_t d[NDim];
    size_t& operator[](size_t i) { return d[i]; }
    size_t operator[](size_t i) const { return d[i]; }
};

template <size_t NDim>
struct hip_tensor_descriptor
{
    hip_tensor_descriptor() = default;
    template <typename T, typename V>
    hip_tensor_descriptor(const T& lens_ext, const V& strides_ext)
    {
        for(size_t i = 0; i < NDim; i++)
            lens[i] = lens_ext[i];
        for(size_t i = 0; i < NDim; i++)
            strides[i] = strides_ext[i];
    }
    hip_index<NDim> multi(size_t idx)
    {
        hip_index<NDim> result{};
        size_t tidx = idx;
        for(size_t is = 0; is < NDim; is++)
        {
            result[is] = tidx / strides[is];
            tidx       = tidx % strides[is];
        }
        return result;
    }
    size_t linear(hip_index<NDim> s)
    {
        size_t idx = 0;
        for(size_t i = 0; i < NDim; i++)
            idx += s[i] * strides[i];
        return idx;
    }
    size_t lens[NDim]    = {};
    size_t strides[NDim] = {};
};

template <typename T, size_t NDim>
__global__ void contiguous_gpu(const T* a,
                               hip_tensor_descriptor<NDim> a_desc,
                               T* at,
                               hip_tensor_descriptor<NDim> at_desc,
                               size_t nelements)
{
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements;
        i += blockDim.x * gridDim.x)
    {
        hip_index<NDim> s = at_desc.multi(i);
        size_t lidx       = a_desc.linear(s);
        at[i]             = a[lidx];
    }
}

void hip_contiguous(migraph::shape output_shape, migraph::argument arg, migraph::argument result)
{
    size_t ndim = output_shape.lens().size();
    visit_all(result, arg)([&](auto output, auto input) {
        if(ndim == 4)
        {
            const auto& s = arg.get_shape();
            hip_tensor_descriptor<4> a_desc(s.lens(), s.strides());
            hip_tensor_descriptor<4> at_desc(output_shape.lens(), output_shape.strides());
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
