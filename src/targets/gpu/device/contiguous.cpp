
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/permutation.hpp>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

__global__ void
cont_kernel(void* in, void* out, int os1, int os2, int os3, int is1, int is2, int is3)
{
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    int i4 = threadIdx.x;

    __half* in_ptr  = reinterpret_cast<__half*>(in);
    __half* out_ptr = reinterpret_cast<__half*>(out);

    int out_idx      = i1 * os1 + i2 * os2 + i3 * os3 + i4;
    int in_idx       = i1 * is1 + i2 * is2 + i3 * is3 + i4;
    out_ptr[out_idx] = in_ptr[in_idx];
}

void contiguous_nonstandard(hipStream_t stream, const argument& result, const argument& arg)
{
    shape s{result.get_shape().type(), result.get_shape().lens()};
    // auto in_s = arg.get_shape();
    // auto perm = find_permutation(in_s);
    // if (in_s.type() == shape::half_type and perm == std::vector<int64_t>({0, 2, 1, 3}))
    // {
    //     auto lens = s.lens();
    //     auto last_dim = s.lens().back();
    //     dim3 grid(lens[0], lens[1], lens[2]);
    //     dim3 block(last_dim);

    //     auto in_stride = in_s.strides();
    //     auto out_stride = s.strides();

    //     cont_kernel<<<grid, block, 0, stream>>>(arg.data(), result.data(), out_stride[0],
    //     out_stride[1], out_stride[2], in_stride[0], in_stride[1], in_stride[2]);
    // }
    // else
    // {
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        hip_visit_views(output_v, input_v, s)([&](auto output, auto input, auto standard_shape) {
            mi_gs_launch(stream,
                         standard_shape)([=](auto idx) __device__ { output[idx] = input[idx]; });
        });
    });
    // }
}

void contiguous_packed(hipStream_t stream, const argument& result, const argument& arg)
{
    index_int nelements = result.get_shape().elements();
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        const auto* input = device_cast(input_v.data());
        auto* output      = device_cast(output_v.data());
        gs_launch(stream, nelements)([=](auto i) __device__ { output[i] = input[i]; });
    });
}

void contiguous(hipStream_t stream, const argument& result, const argument& arg)
{
    if(result.get_shape() == arg.get_shape() and result.get_shape().packed())
        contiguous_packed(stream, result, arg);
    else
        contiguous_nonstandard(stream, result, arg);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
