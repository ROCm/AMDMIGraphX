#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/device/fast_div.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)
void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
{
    auto relements = arg1.get_shape().lens().back();
    assert(relements <= 1024);
    auto nelements    = result.get_shape().elements() / relements;
    auto input_shape  = arg1.get_shape();
    auto output_shape = result.get_shape();
    auto reduce_output_lens(output_shape.lens());
    reduce_output_lens.back() = 1;

    std::vector<index_int> reduce_lens = get_reduce_lens(input_shape.lens(), reduce_output_lens);

    hip_visit_all(result, arg1)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 1024;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = fast_div(i, block_size_div);
            const auto base_idx = out_idx * relements;
            value_type x = input.data()[base_idx];

            auto m = block_reduce<max_block_size>(
                         idx, sum{}, 0, relements, [&](auto) { return x; }) /
                     relements;

            x = x - m;

            auto r = block_reduce<max_block_size>(
                         idx, sum{}, 0, relements, [&](auto) { return x*x; }) /
                     relements;

            const auto eps = ::rsqrt(r + 1e-12);
            output.data()[base_idx] = x * eps;

        });

    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
