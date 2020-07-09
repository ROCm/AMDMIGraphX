#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
{
    auto relements    = arg1.get_shape().lens().back();
    auto nelements    = result.get_shape().elements() / relements;
    auto input_shape  = arg1.get_shape();
    auto output_shape = result.get_shape();
    auto reduce_output_lens(output_shape.lens());
    reduce_output_lens.back() = 1;

    std::vector<index_int> reduce_lens = get_reduce_lens(input_shape.lens(), reduce_output_lens);
    shape reduce_slice{output_shape.type(), reduce_lens};
    shape reduce_output_shape{output_shape.type(), reduce_output_lens};

    hip_visit_all(result, arg1)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = i / block_size;
            const auto base_idx = out_idx * relements;
            value_type x[4];

            idx.local_stride(relements,
                             [&](auto j) __device__ { x[j-idx.local] = input.data()[base_idx + j]; });


            auto m = block_reduce<max_block_size>(
                         idx, sum{}, 0, relements, [&](auto j) __device__ { return x[j-idx.local]; }) / relements;

            idx.local_stride(relements, [&](auto j) __device__ { x[j-idx.local] = x[j-idx.local] - m; });

            auto r = block_reduce<max_block_size>(
                         idx,
                         sum{},
                         0,
                         relements,
                         [&](auto j) __device__ { return ::pow(to_hip_type(x[j-idx.local]), 2); }) /
                     relements;

            idx.local_stride(relements, [&](auto j) __device__ {
                output.data()[base_idx + j] = (x[j-idx.local]) * ::rsqrt(r + 1e-12);
            });

        });

    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
