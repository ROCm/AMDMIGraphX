
#include <migraph/miopen/hip.hpp>

#include <migraph/manage_ptr.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraph {
namespace miopen {

using hip_ptr = MIGRAPH_MANAGE_PTR(void, hipFree);

template <int NDIM> 
struct HIPTensorDescriptor 
{ 
    size_t lens[NDIM];
    size_t strides[NDIM]; 
};

template <typename T, int NDIM>
__global__
void contiguous_gpu(const T* A,
                    HIPTensorDescriptor<NDIM> td_a,
                    T* At,
                    HIPTensorDescriptor<NDIM> td_at,
                    size_t nelements) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < nelements; i += blockDim.x * gridDim.x) {
    size_t s[NDIM];
    multiindex<NDIM>(td_at.strides, i, s);
    size_t lidx = 0;
    for (size_t j = 0; j < NDIM; j++) lidx += s[j] * td_a.strides[j];
    At[i] = A[lidx];
  }
}

hip_ptr allocate_gpu(std::size_t sz)
{
    void* result;
    // TODO: Check status
    hipMalloc(&result, sz);
    return hip_ptr{result};
}

template <class T>
hip_ptr write_to_gpu(const T& x)
{
    using type = typename T::value_type;
    auto size  = x.size() * sizeof(type);
    return write_to_gpu(x.data(), size);
}

template <class T>
std::vector<T> read_from_gpu(const void* x, std::size_t sz)
{
    std::vector<T> result(sz);
    // TODO: Check status
    hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    return result;
}

hip_ptr write_to_gpu(const void* x, std::size_t sz)
{
    auto result = allocate_gpu(sz);
    // TODO: Check status
    hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    return result;
}

migraph::argument allocate_gpu(migraph::shape s)
{
    auto p = share(allocate_gpu(s.bytes()));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

migraph::argument to_gpu(migraph::argument arg)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes()));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

migraph::argument from_gpu(migraph::argument arg)
{
    migraph::argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

migraph::argument hip_contiguous(migraph::argument arg, migraph::shape output_shape) 
{
    migraph::argument result{output_shape};
    visit_all(result, arg)([&](auto output, auto input) {
        HIPTensorDescriptor td_a, td_at;
        auto s = arg.get_shape();
        for (int i = 0; i < output_shape.lens().size(); i++) {
          td_a.strides[i] = s.strides().at(i);
          td_at.strides[i] = output_shape.strides().at(i);
        }
        dim3 nblocks(512);
        dim3 nthreads(512);
        hipLaunchKernelGGL((contiguous_gpu<int, 4>), nblocks, nthreads, 0, 0, 
                     input.data(),
                     td_a,
                     output.data(),
                     td_at,
                     s.elements());        
    });
    return result;
}

} // namespace miopen

} // namespace migraph
