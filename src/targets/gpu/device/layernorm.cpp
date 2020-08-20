#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/device/fast_div.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#define BLOCK_SIZE 1024

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

    // std::cout << "nelements: " << nelements << ", " << nelements_v << std::endl;
    // std::cout << "relements: " << relements << ", " << relements_v << std::endl;

#if BLOCK_SIZE == 1024
    hip_visit_all(result, arg1)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 1024;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx   = fast_div(i, block_size_div);
            const auto base_idx  = out_idx * relements;
            const auto input_idx = base_idx + idx.local;
            const bool in_range  = idx.local < relements;

            auto mean = [&](auto z) {
                return block_reduce<max_block_size>(
                           idx, sum{}, value_type(0), relements, [=](auto) { return z; }) /
                       relements;
            };

            // m = x - mean(x)
            value_type x = in_range ? input.data()[input_idx] : 0;
            value_type m = x - mean(x);

            // mean(m ^ 2) + 1e-12
            value_type r = mean(m * m) + 1e-12;

            // m * rsqrt(mean(m ^ 2) + 1e-12)
            if(in_range)
                output.data()[input_idx] = m * ::rsqrt(r);
        });

    });
#elif BLOCK_SIZE == 256
    hip_visit_all(result, arg1)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = fast_div(i, block_size_div);
            const auto base_idx = out_idx * relements;
            value_type x_data[4];
            auto with_x = [&](auto f) {
                int offset = 0;
                return [=, &x_data](auto j) mutable { return f(x_data[offset++], j); };
            };

            idx.local_stride(relements,
                             with_x([&](auto& x, auto j) { x = input.data()[base_idx + j]; }));

            auto m = block_reduce<max_block_size>(
                         idx, sum{}, 0, relements, with_x([](auto& x, auto) { return x; })) /
                     relements;

            idx.local_stride(relements, with_x([&](auto& x, auto) { x = x - m; }));

            auto r = block_reduce<max_block_size>(
                         idx, sum{}, 0, relements, with_x([&](auto& x, auto) { return x * x; })) /
                     relements;

            const auto eps = ::rsqrt(r + 1e-12);
            idx.local_stride(
                relements, with_x([&](auto& x, auto j) { output.data()[base_idx + j] = x * eps; }));

        });
    });
#else

    hip_vec_visit_all<vec_size>(result, arg1)([&](auto output, auto input) {
        std::cout << "******************** hip_vec_visit_all ********************" << std::endl;
        using value_type = typename decltype(input)::value_type;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements_v, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);
        const auto ielements_v           = result.get_shape().elements() / vec_size;
        assert(relements_v < block_size);

        std::cout << "block_size: " << block_size << std::endl;
        std::cout << "relements: " << relements << std::endl;
        std::cout << "relements_v: " << relements_v << std::endl;
        std::cout << "nelements: " << nelements << std::endl;
        std::cout << "nelements_v: " << nelements_v << std::endl;

        for(int global = 0; global < nelements_v; global++)
        {
            std::cout << "global: " << global << std::endl;
            for(int local = 0; local < block_size; local++)
            {
                int i                = global * block_size + local;
                const auto out_idx   = fast_div(i, block_size_div);
                const auto base_idx  = out_idx * relements_v;
                const auto input_idx = base_idx + local;
                const bool in_range  = input_idx < ielements_v;
                if(local >= relements_v)
                    continue;
                std::cout << "input_idx: " << input_idx << " {";
                std::cout << "i=" << i << ", ";
                std::cout << "base_idx=" << base_idx << ", ";
                std::cout << "out_idx=" << out_idx << ", ";
                std::cout << "}" << std::endl;
            }
        }

        gs_launch(stream, nelements_v * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx   = fast_div(i, block_size_div);
            const auto base_idx  = out_idx * relements_v;
            const auto input_idx = base_idx + idx.local;
            const bool in_range  = idx.local < relements_v;

            auto mean = [&](auto z) -> value_type {
                return block_reduce<max_block_size>(
                           idx, sum{}, value_type(0), relements_v, [=](auto) { return z; }) /
                       value_type(relements);
            };

            // m = x - mean(x)
            value_type x = in_range ? input.data()[input_idx] : 0;
            value_type m = x - mean(x);

            // mean(m ^ 2) + 1e-12
            value_type r = mean(m * m) + value_type(1e-12);

            // rsqrt(mean(m ^ 2) + 1e-12)
            value_type d;
            for(index_int k = 0; k < vec_size; k++)
                d[k] = ::rsqrt(r[k]);
            // m * rsqrt(mean(m ^ 2) + 1e-12)
            if(in_range)
                output.data()[input_idx] = m * d;
        });

    });
#endif
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
