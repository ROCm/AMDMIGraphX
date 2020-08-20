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
    const index_int vec_size           = 4;
    assert((nelements % vec_size) == 0);
    assert((relements % vec_size) == 0);
    auto nelements_v = nelements / vec_size;
    auto relements_v = relements / vec_size;

    hip_vec_visit_all<vec_size>(result, arg1)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements_v, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);

        gs_launch(stream, nelements_v * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = fast_div(i, block_size_div);
            const auto base_idx = out_idx * relements_v;

            value_type x = input.data()[base_idx];

            auto m =
                block_reduce<max_block_size>(idx, sum{}, 0, relements_v, [&](auto) { return x; }) /
                relements;

            x = x - m;

            auto r = (block_reduce<max_block_size>(
                          idx, sum{}, 0, relements, [&](auto) { return x * x; }) /
                      relements) +
                     value_type(1e-12);

            value_type eps;
            for(index_int k = 0; k < vec_size; k++)
                eps[k] = ::rsqrt(r[k]);
            output.data()[base_idx] = x * eps;
        });

    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
