
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_REDUCE_HPP

#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/gpu/device/multi_index.hpp>

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

struct mean
{
    size_t item_num = 1;
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return static_cast<T>(x / item_num);
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
template <index_int N, class Op, class T, class F>
__device__ auto block_reduce(index idx, Op op, T init, index_int n, F f)
{
    using type = decltype(f(idx.local));
    MIGRAPHX_DEVICE_SHARED type buffer[N];
    type x = init;
    idx.local_stride(n, [&](auto i) { x = op(x, f(i)); });
    buffer[idx.local] = x;
    __syncthreads();

    for(index_int s = 1; s < idx.nlocal(); s *= 2)
    {
        const index_int index = 2 * s * idx.local;
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
    static const index_int n = sizeof(T) < 4 ? 1 : sizeof(T) / 4;
    union type
    {
        uint32_t reg[n];
        T data;
    };
    type output{};
    type input{};
    // cppcheck-suppress unreadVariable
    input.data = x;
    for(index_int i = 0; i < n; i++)
    {
        output.reg[i] = __llvm_amdgcn_move_dpp(input.reg[i], DppCtrl, RowMask, BankMask, BoundCtrl);
    }
    return output.data;
}

template <class T, class Op>
__device__ void dpp_reduce(T& in, Op op)
{
    T out{};
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
#if defined(MIGRAPHX_USE_CLANG_TIDY) || defined(CPPCHECK)
    x = 1;
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

template <index_int N,
          class Op,
          class T,
          class ForStride,
          class F,
          MIGRAPHX_REQUIRES(not std::is_integral<ForStride>{})>
__device__ auto block_reduce(index idx, Op op, T init, ForStride fs, F f)
{
    using type = decltype(f(deduce_for_stride(fs)));
    MIGRAPHX_DEVICE_SHARED type buffer[N / 64];
    type x = init;
    fs([&](auto i) { x = op(x, f(i)); });
    dpp_reduce(x, op);

    const auto ldsidx = idx.local / 64;
    if((idx.local % 64) == 63)
    {
        buffer[ldsidx] = x;
    }
    __syncthreads();

    type y = init;
    for(index_int i = 0; i < idx.nlocal() / 64; i++)
    {
        y = op(y, buffer[i]);
    }
    return y;
}

template <index_int N, class Op, class T, class F>
__device__ auto block_reduce(index idx, Op op, T init, index_int n, F f)
{
    auto midx = make_multi_index(idx.local, idx.nlocal());
    // Workaround hcc, create a local array
    auto fs = midx.id;
    fs[0]   = n;
    return block_reduce<N>(
        idx, op, init, midx.for_stride(fs), [&](auto mi) __device__ { return f(mi[0]); });
}

#endif
constexpr index_int compute_block_size(index_int n, index_int max_block_size)
{
    size_t block_size = 64;
    while(block_size < max_block_size and block_size < n)
        block_size *= 2;
    return block_size;
}

template <class Op, class T, class Input, class Output>
void reduce_multi_impl(hipStream_t stream,
                       const argument& result,
                       const argument& arg,
                       Op op,
                       T init,
                       Input read_input,
                       Output read_output,
                       const shape& reduce_slice)
{
    hip_visit_all(result, arg, reduce_slice)([&](auto output, auto input, auto reduce_shape) {
        auto nelements = result.get_shape().elements();
        auto relements = reduce_slice.elements();

        const index_int max_block_size = 256;
        const index_int block_size     = compute_block_size(relements, max_block_size);
        mi_launch(stream, output.get_shape(), reduce_shape, block_size)([=](auto idx, auto global, auto local) __device__ {
            global([&](auto i) __device__ {
                auto r = block_reduce<max_block_size>(idx, op, init, local, [&](auto j) __device__ {
                    return read_input(input[i + j]);
                });
                if(idx.local == 0)
                    output[i] = read_output(r);
            });
        });
    });
}

template <class Op, class T, class Input, class Output>
void reduce_standard_impl(hipStream_t stream,
                          const argument& result,
                          const argument& arg,
                          Op op,
                          T init,
                          Input read_input,
                          Output read_output,
                          index_int relements)
{
    hip_visit_all(result, arg)([&](auto output, auto input) {
        auto nelements = result.get_shape().elements();

        const index_int max_block_size = 256;
        const index_int block_size     = compute_block_size(relements, max_block_size);
        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = i / block_size;
            const auto base_idx = out_idx * relements;
            auto r = block_reduce<max_block_size>(idx, op, init, relements, [&](auto j) __device__ {
                return read_input(input.data()[base_idx + j]);
            });
            if(idx.local == 0)
                output.data()[out_idx] = read_output(r);
        });
    });
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
    assert(output_shape.lens().size() == input_shape.lens().size());
    if(input_shape.standard() and output_shape.standard() and
       output_shape.lens().back() != input_shape.lens().back() and
       std::equal(output_shape.lens().begin(),
                  std::prev(output_shape.lens().end()),
                  input_shape.lens().begin()))
    {
        reduce_standard_impl(
            stream, result, arg, op, init, read_input, read_output, input_shape.lens().back());
    }
    else
    {
        std::vector<index_int> reduce_lens;
        std::transform(output_shape.lens().begin(),
                       output_shape.lens().end(),
                       input_shape.lens().begin(),
                       std::back_inserter(reduce_lens),
                       [](auto x, auto y) -> index_int {
                           if(x == y)
                               return 1;
                           else
                               return y;
                       });
        shape reduce_slice{output_shape.type(), reduce_lens};
        reduce_multi_impl(stream, result, arg, op, init, read_input, read_output, reduce_slice);
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
