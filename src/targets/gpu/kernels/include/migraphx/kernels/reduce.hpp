/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
#define MIGRAPHX_GUARD_KERNELS_REDUCE_HPP

#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

#if MIGRAPHX_HAS_DPP

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
#if __AMDGCN_WAVEFRONT_SIZE == 64
    out = dpp_mov<dpp_row_bcast(15), 0xa>(in);
    in  = op(in, out);
    out = dpp_mov<dpp_row_bcast(31), 0xc>(in);
    in  = op(in, out);
#endif
}
#if defined(MIGRAPHX_USE_CLANG_TIDY) || defined(CPPCHECK)
// NOLINTNEXTLINE
#define MIGRAPHX_DPP_REDUCE_ASM(x, ins) x = 1
#elif __AMDGCN_WAVEFRONT_SIZE == 64
#define MIGRAPHX_DPP_REDUCE_ASM(x, ins)                                       \
    __asm__ volatile("s_nop 4\n" #ins " %0 %0 %0 row_shr:1\n"                 \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:2\n"                 \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:4 bank_mask:0xe\n"   \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:8 bank_mask:0xc\n"   \
                     "s_nop 1\n" #ins " %0 %0 %0 row_bcast:15 row_mask:0xa\n" \
                     "s_nop 1\n" #ins " %0 %0 %0 row_bcast:31 row_mask:0xc\n" \
                     "s_nop 1\n"                                              \
                     : "=v"(x)                                                \
                     : "0"(x))
#else
#define MIGRAPHX_DPP_REDUCE_ASM(x, ins)                                     \
    __asm__ volatile("s_nop 4\n" #ins " %0 %0 %0 row_shr:1\n"               \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:2\n"               \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:4 bank_mask:0xe\n" \
                     "s_nop 1\n" #ins " %0 %0 %0 row_shr:8 bank_mask:0xc\n" \
                     "s_nop 1\n"                                            \
                     "s_nop 1\n"                                            \
                     : "=v"(x)                                              \
                     : "0"(x))
#endif

// NOLINTNEXTLINE
#define MIGRAPHX_DPP_REDUCE(op, prefix)                                                            \
    __device__ inline void dpp_reduce(double& x, op) { MIGRAPHX_DPP_REDUCE_ASM(x, prefix##_f64); } \
    __device__ inline void dpp_reduce(float& x, op) { MIGRAPHX_DPP_REDUCE_ASM(x, prefix##_f32); }  \
    __device__ inline void dpp_reduce(half& x, op) { MIGRAPHX_DPP_REDUCE_ASM(x, prefix##_f16); }   \
    __device__ inline void dpp_reduce(int32_t& x, op)                                              \
    {                                                                                              \
        MIGRAPHX_DPP_REDUCE_ASM(x, prefix##_u32);                                                  \
    }                                                                                              \
    __device__ inline void dpp_reduce(uint32_t& x, op) { MIGRAPHX_DPP_REDUCE_ASM(x, prefix##_u32); }

MIGRAPHX_DPP_REDUCE(op::sum, v_add)
MIGRAPHX_DPP_REDUCE(op::max, v_max)
MIGRAPHX_DPP_REDUCE(op::min, v_min)
MIGRAPHX_DPP_REDUCE(op::product, v_mul)

template <class Op, class T, class Index, class F>
__device__ auto block_reduce(index idx, Op op, T init, Index n, F f)
{
    MIGRAPHX_ASSERT(idx.max_nlocal() == idx.nlocal());
#if __AMDGCN_WAVEFRONT_SIZE == 32
    constexpr index_int lanes_per_thread = 16;
#else
    constexpr index_int lanes_per_thread = 64;
#endif
    using type = decltype(f(0));
    __shared__ type buffer[idx.max_nlocal() / lanes_per_thread];
    type x = init;
    idx.local_stride(n, [&](auto i) { x = op(x, f(i)); });
    dpp_reduce(x, op);

    const auto ldsidx = idx.local / lanes_per_thread;
    if((idx.local % lanes_per_thread) == lanes_per_thread - 1)
    {
        buffer[ldsidx] = x;
    }
    __syncthreads();

    type y = init;
    for(index_int i = 0; i < idx.nlocal() / lanes_per_thread; i++)
    {
        y = op(y, buffer[i]);
    }
    return y;
}
#else
template <class Op, class T, class Index, class F>
__device__ auto block_reduce(index idx, Op op, T init, Index n, F f)
{
    MIGRAPHX_ASSERT(idx.max_nlocal() == idx.nlocal());
    using type = decltype(f(0));
    __shared__ type buffer[idx.max_nlocal()];
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
#endif

template <class Output, class Input, class T>
constexpr auto reduce_slice(Input input, T i)
{
    constexpr auto lens = transform(get_shape_c<Input>{}.lens,
                                    get_shape_c<Output>{}.lens,
                                    [](index_int x, index_int y) -> index_int {
                                        if(x == y)
                                            return 1;
                                        return x;
                                    });
    ;
    constexpr auto s = make_shape(lens, get_shape_c<Input>{}.strides);
    MIGRAPHX_ASSERT((input.get_shape().index(i) + s.element_space()) <=
                    input.get_shape().element_space());
    return make_tensor_view(&input[i], s);
}

namespace reduce {

template <class Slicer, class F>
constexpr auto sliced(Slicer slicer, F f)
{
    return [=](auto x, auto... xs) {
        // TODO: assert all elements are the same
        return f(slicer(x), slicer(xs)...);
    };
}

template <class Input, index_int Axis>
constexpr auto compute_reduce_axis()
{
    constexpr auto lens =
        transform_i(get_shape_c<Input>{}.lens, [](index_int x, index_int i) -> index_int {
            if(i == Axis)
                return 1;
            return x;
        });
    return make_shape(lens, get_shape_c<Input>{}.strides);
}

template <class Input, index_int Axis>
using with_axis = decltype(compute_reduce_axis<Input, Axis>());

struct block
{
    template <class Slicer>
    struct reducer
    {
        index idx;
        Slicer slice;
        template <class Op, class T, class Read>
        __device__ auto reduce(Op op, T init, Read read) const
        {
            return sliced(slice, [=](auto x, auto... xs) {
                return block_reduce(idx, op, init, x.get_shape().elements(), [&](auto j) {
                    return vec_reduce(read(x[j], xs[j]...), op);
                });
            });
        }

        template <class F>
        __device__ void outer(F f) const
        {
            if(idx.local == 0)
                f();
        }

        template <class F>
        __device__ auto inner(F f) const
        {
            return sliced(slice, [=](auto x, auto... xs) {
                idx.local_stride(x.get_shape().elements(), [&](auto j) { f(x[j], xs[j]...); });
            });
        }

        template <class Input>
        constexpr auto elements() const
        {
            using reduce_type        = decltype(slice(Input{}));
            using value_type         = typename Input::type;
            constexpr auto relements = get_shape_c<reduce_type>{}.elements();
            if constexpr(vec_size<value_type>() > 1)
                return relements * vec_size<value_type>();
            else
                return relements;
        }
    };

    template <class Slicer>
    static __device__ auto make(index idx, Slicer slicer)
    {
        return reducer<Slicer>{idx, slicer};
    }

    template <class Output, class F>
    static __device__ void run(F f)
    {
        auto idx                 = make_index();
        constexpr auto nelements = get_shape_c<Output>{}.elements();
        idx.global_stride(nelements * idx.nlocal(), [&](auto i) {
            const auto out_idx = get_shape_c<Output>{}.multi(i / idx.nlocal());
            f(out_idx, make(idx, [&](auto input) { return reduce_slice<Output>(input, out_idx); }));
        });
    }
};

struct lane
{
    template <class Slicer>
    struct reducer
    {
        index idx;
        Slicer slice;
        template <class Op, class T, class Read>
        __device__ auto reduce(Op op, T init, Read read) const
        {
            return sliced(slice, [=](auto x, auto... xs) {
                using type = typename decltype(x)::type;
                type r     = init;
                for(index_int j = 0; j < x.get_shape().elements(); j++)
                {
                    r = op(r, read(x[j], xs[j]...));
                }
                return r;
            });
        }

        template <class F>
        __device__ void outer(F f) const
        {
            f();
        }

        template <class F>
        __device__ auto inner(F f) const
        {
            return sliced(slice, [=](auto x, auto... xs) {
                for(index_int j = 0; j < x.get_shape().elements(); j++)
                {
                    f(x[j], xs[j]...);
                }
            });
        }

        template <class Input>
        constexpr auto elements() const
        {
            using reduce_type = decltype(slice(Input{}));
            return get_shape_c<reduce_type>{}.elements();
        }
    };

    template <class Slicer>
    static __device__ auto make(index idx, Slicer slicer)
    {
        return reducer<Slicer>{idx, slicer};
    }

    template <class Output, class F>
    static __device__ void run(F f)
    {
        auto idx                 = make_index();
        constexpr auto nelements = get_shape_c<Output>{}.elements();
        idx.global_stride(nelements, [&](auto i) {
            const auto out_idx = get_shape_c<Output>{}.multi(i);
            f(out_idx, make(idx, [&](auto input) { return reduce_slice<Output>(input, out_idx); }));
        });
    }
};

} // namespace reduce

template <class Algo,
          class Op,
          class T,
          class Input,
          class Output,
          class ReadInput,
          class WriteOuput>
__device__ void
simple_reduce(Op op, T init, Input input, Output output, ReadInput read, WriteOuput write)
{
    Algo::template run<Output>([&](auto out_idx, auto r) {
        auto x = r.reduce(op, init, read)(input);
        r.outer([&] { output[out_idx] = write(x); });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_REDUCE_HPP
