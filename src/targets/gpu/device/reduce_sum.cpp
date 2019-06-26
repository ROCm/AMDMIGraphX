#include <migraphx/gpu/device/reduce_sum.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

struct sum
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T operator()(T x, T y) const
    {
        return x + y;
    }
};

template <std::size_t N, class Op, class T, class F>
__device__ auto block_reduce(index idx, Op op, T init, std::size_t n, F f)
{
    using type = decltype(f(idx.local));
    MIGRAPHX_DEVICE_SHARED type buffer[N];
    type x = init;
    idx.local_stride(n, [&](auto i) { x = op(x, f(i)); });
    buffer[idx.local] = x;
    __syncthreads();

    for(std::size_t s = 1; s < idx.nlocal(); s *= 2)
    {
        const std::size_t index = 2 * s * idx.local;
        if(index < idx.nlocal())
        {
            buffer[index] = op(buffer[index], buffer[index + s]);
        }
        __syncthreads();
    }
    return buffer[0];
}

constexpr std::size_t compute_block_size(std::size_t n, std::size_t max_block_size)
{
    size_t block_size = 1;
    while(block_size < max_block_size and block_size < n)
        block_size *= 2;
    return block_size;
}

void reduce_sum(hipStream_t stream, const argument& result, const argument& arg)
{
    auto&& output_shape = result.get_shape();
    auto&& input_shape  = arg.get_shape();
    std::vector<std::size_t> reduce_lens;
    std::transform(output_shape.lens().begin(),
                   output_shape.lens().end(),
                   input_shape.lens().begin(),
                   std::back_inserter(reduce_lens),
                   [](auto x, auto y) -> std::size_t {
                       if(x == y)
                           return 1;
                       else
                           return y;
                   });
    shape reduce_slice{output_shape.type(), reduce_lens};
    hip_visit_all(result, arg, reduce_slice)([&](auto output, auto input, auto reduce_shape) {
        auto nelements = result.get_shape().elements();
        auto relements = reduce_slice.elements();

        const std::size_t max_block_size = 1024;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx = i / block_size;
            auto base_idx = output.get_shape().multi(out_idx);
            auto r = block_reduce<max_block_size>(idx, sum{}, 0, relements, [&](auto j) __device__ {
                auto reduce_idx = reduce_shape.multi(j);
                return input[reduce_idx + base_idx];
            });
            if(idx.local == 0)
                output.data()[out_idx] = r;
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
