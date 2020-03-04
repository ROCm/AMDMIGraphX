#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
void layernorm(hipStream_t stream, const argument& result, const argument& arg1
               //    ,
               //    const argument& arg2,
               //    const argument& arg3
)
{
    auto relements    = arg1.get_shape().lens().back();
    auto input_shape  = arg1.get_shape();
    auto output_shape = result.get_shape();
    auto reduce_output_lens(output_shape.lens());
    reduce_output_lens.back() = 1;

    std::vector<index_int> reduce_lens = get_reduce_lens(input_shape.lens(), reduce_output_lens);
    shape reduce_slice{output_shape.type(), reduce_lens};
    shape reduce_output_shape{output_shape.type(), reduce_output_lens};
    hip_visit_all(result, arg1, reduce_slice, reduce_output_shape)(
        [&](auto output, auto input, auto reduce_shape, auto reduce_output) {
            using value_type = typename decltype(input)::value_type;

            const std::size_t max_block_size = 256;
            const std::size_t block_size     = compute_block_size(relements, max_block_size);

            mi_launch(stream, reduce_output, reduce_shape, block_size)(
                [=](auto idx, auto global, auto local) __device__ {
                    global([&](auto i) __device__ {

                        value_type x[4];
                        int k = 0;
                        local([&](auto j) {
                            k++;
                            x[k - 1] = input[i + j];
                        });

                        k      = 0;
                        auto m = block_reduce<max_block_size>(idx,
                                                              sum{},
                                                              0,
                                                              relements,
                                                              [&](auto) __device__ {
                                                                  k++;
                                                                  return x[k - 1];
                                                              }) /
                                 relements;

                        k = 0;
                        local([&](auto) {
                            k++;
                            x[k - 1] -= m;
                        });

                        k = 0;
                        auto r =
                            block_reduce<max_block_size>(idx,
                                                         sum{},
                                                         0,
                                                         relements,
                                                         [&](auto) __device__ {
                                                             k++;
                                                             return ::pow(to_hip_type(x[k - 1]), 2);
                                                         }) /
                            relements;

                        k = 0;
                        local([&](auto j) {
                            k++;
                            output[i + j] = x[k - 1] * ::rsqrt(r + 1e-12);
                            // auto a        = scale[i + j] * ::rsqrt(r + 1e-12);
                            // output[i + j] = a * x[k - 1] + bias[i + j];
                        });

                    });
                });
        });

    // hip_visit_all(result, arg1, arg2, arg3, reduce_slice, reduce_output_shape)(
    // [&](auto output, auto input, auto scale, auto bias, auto reduce_shape, auto reduce_output) {
    //     using value_type = typename decltype(input)::value_type;
    //      auto nelements   = result.get_shape().elements() / relements;

    //     const std::size_t max_block_size = 256;
    //     const std::size_t block_size     = compute_block_size(relements, max_block_size);

    //     gs_launch(stream, nelements * block_size, block_size)(
    //         [=](auto i, auto idx) __device__ {
    //                 const auto out_idx  = i / block_size;
    //                 const auto base_idx = out_idx * relements;
    //                 value_type x[4];
    //                 int k = 0;

    //                 idx.local_stride(relements, [&](auto j) {
    //                     x[j] = input.data()[base_idx + j];
    //                 });

    //                 k      = 0;
    //                 auto m = block_reduce<max_block_size>(idx,
    //                                                       sum{},
    //                                                       0,
    //                                                       relements,
    //                                                       [&](auto j) __device__ {
    //                                                           return x[j];
    //                                                       }) /
    //                          relements;

    //                 k = 0;

    //                 idx.local_stride(relements, [&](auto j) {
    //                     x[j] -= m;
    //                 });

    //                 k = 0;
    //                 auto r =
    //                     block_reduce<max_block_size>(idx,
    //                                                  sum{},
    //                                                  0,
    //                                                  relements,
    //                                                  [&](auto j) __device__ {
    //                                                     //  k++;
    //                                                      return ::pow(to_hip_type(x[j]), 2);
    //                                                  }) /
    //                     relements;

    //                 k = 0;
    //                 idx.local_stride(relements, [&](auto j) {
    //                     // k++;

    //                     auto a        = scale[i + j] * ::rsqrt(r + 1e-12);
    //                     output.data()[base_idx + j] = a * x[j] + bias[i + j];
    //                     // output[i + j] = a * x[k - 1] + bias[i + j];
    //                 });

    //         });
    // });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
