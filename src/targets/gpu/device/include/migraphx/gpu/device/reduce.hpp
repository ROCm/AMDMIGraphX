
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_HPP

#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

struct sum
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x + y;
    }
};

struct id
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return x;
    }
};

struct max
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x > y ? x : y;
    }
};

struct min
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x < y ? x : y;
    }
};

struct lowest
{
    template <class T>
    operator T() const
    {
        return device_cast(std::numeric_limits<host_type<T>>::lowest());
    }
};

struct highest
{
    template <class T>
    operator T() const
    {
        return device_cast(std::numeric_limits<host_type<T>>::max());
    }
};

#ifdef MIGRAPHX_NO_DPP
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
        if(index + s < idx.nlocal())
        {
            buffer[index] = op(buffer[index], buffer[index + s]);
        }
        __syncthreads();
    }
    return buffer[0];
}
#else
constexpr unsigned int dpp_row_shr(unsigned int x) { return 0x110u | x; }

constexpr unsigned int dpp_row_bcast(unsigned int x)
{
    unsigned int y = 0;
    switch(x)
    {
    case 15: y = 0x142; break;
    case 31: y = 0x143; break;
    default: throw std::runtime_error("Unknown bcast");
    }
    return y;
}

template <unsigned int DppCtrl,
          unsigned int RowMask  = 0xf,
          unsigned int BankMask = 0xf,
          bool BoundCtrl        = false,
          class T>
__device__ T dpp_mov(T& x)
{
    static const std::size_t n = sizeof(T) < 4 ? 1 : sizeof(T) / 4;
    union type
    {
        uint32_t reg[n];
        T data;
    };
    type output{};
    type input{};
    // cppcheck-suppress unreadVariable
    input.data = x;
    for(std::size_t i = 0; i < n; i++)
    {
        output.reg[i] = __llvm_amdgcn_move_dpp(input.reg[i], DppCtrl, RowMask, BankMask, BoundCtrl);
    }
    return output.data;
}

template <class T, class Op>
__device__ void dpp_reduce(T& in, Op op)
{
    T out;
    out = dpp_mov<dpp_row_shr(1)>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_shr(2)>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_shr(4), 0xf, 0xe>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_shr(8), 0xf, 0xc>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_bcast(15), 0xa>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_bcast(31), 0xc>(in);
    in  = op(in, out);
}

__device__ inline void dpp_reduce(float& x, sum)
{
#ifdef MIGRAPHX_USE_CLANG_TIDY
    (void)x;
#else
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:2\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "s_nop 1\n"
                     : "=v"(x)
                     : "0"(x));
#endif
}

template <std::size_t N, class Op, class T, class F>
__device__ auto  block_reduce(index idx, Op op, T init, std::size_t n, F f) 
{
    using type = decltype(f(idx.local));
    MIGRAPHX_DEVICE_SHARED type buffer[N / 64];
    type x = init;
    idx.local_stride(n, [&](auto i) { x = op(x, f(i)); });
    dpp_reduce(x, op);

    const auto ldsidx = idx.local / 64;
    if((idx.local % 64) == 63)
    {
        buffer[ldsidx] = x;
    }
    __syncthreads();

    type y = 0;
    for(std::size_t i = 0; i < idx.nlocal() / 64; i++)
    {
        y = op(y, buffer[i]);
    }
    return y;
}
#endif
constexpr std::size_t compute_block_size(std::size_t n, std::size_t max_block_size)
{
    size_t block_size = 64;
    while(block_size < max_block_size and block_size < n)
        block_size *= 2;
    return block_size;
}

template <class Op, class T, class Input, class Output>
void reduce(hipStream_t stream,
            const argument& result,
            const argument& arg,
            Op op,
            T init,
            Input read_input,
            Output read_output)
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
            auto base_idx      = output.get_shape().multi(out_idx);
            auto r = block_reduce<max_block_size>(idx, op, init, relements, [&](auto j) __device__ {
                auto reduce_idx = reduce_shape.multi(j);
                return read_input(input[reduce_idx + base_idx]);
            });
            if(idx.local == 0)
                output.data()[out_idx] = read_output(r);
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
