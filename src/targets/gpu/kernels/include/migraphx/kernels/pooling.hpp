#ifndef MIGRAPHX_GUARD_KERNELS_POOLING_HPP
#define MIGRAPHX_GUARD_KERNELS_POOLING_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/array.hpp>

namespace migraphx {

struct max_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return lowest{}; }

    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y)
    {
        return max(x, y);
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int)
    {
        return x;
    }
};

struct average_pool
{
    MIGRAPHX_DEVICE_CONSTEXPR auto init() { return 0.0; }

    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y)
    {
        return x + y;
    }

    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR T final(T x, index_int y)
    {
        return (y == 0) ? T{0.0} : T{x / y};
    }
};

template<class Window, class Stride, class Padding>
struct window
{
    Window win = {};
    Stride stride = {};
    Padding padding = {};

    using rank = decltype(Window{}.size());

    constexpr auto size() const
    {
        return return_c([] { return Window{}.product(); });
    }

    constexpr auto has_padding() const
    {
        return return_c([] { return Padding{} == 0; });
    }

    template<class Index, class F>
    constexpr void visit(Index i, F f) const
    {
        auto win_start = generate_array<diff_int>(rank{}, [&](auto j) {
            diff_int w = win[j];
            diff_int dim = i[j];
            MIGRAPHX_ASSERT(w >= 1);
            if(w == 1)
                return dim;
            diff_int s = stride[j];
            diff_int p = padding[j];
            return (dim * s) - p;
        });
        repeat(size(), [&](auto j) {
            f(win_start + win.multi(j));
        });
    }
};

template<class Window, class Stride, class Padding>
constexpr window<Window, Stride, Padding> make_window(Window w, Stride s, Padding p)
{
    return {w, s, p};
}


template<class Op, class Window, class Output, class Input>
__device__ void pooling(Op op, Window w, Output output, Input input)
{
    auto idx = make_index();
    using type = typename Output::type;
    idx.global_stride(output.get_shape().elements(), [&](auto i) {
        auto out_idx = output.get_shape().multi(i);
        index_int pool_size = w.size();
        type x = op.init();
        w.visit(out_idx, [&](auto j) {
            if (j < input.get_shape().lens)
            {
                x = op(x, input[j]);
            }
            else
            {
                // TODO: Only do for lpnorm
                // x = op(x, op.init());
                if constexpr(is_same<Op, average_pool>{})
                {
                    pool_size--;
                }
            }
        });
        output[out_idx] = op.final(x, pool_size);
    });
}


} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POOLING_HPP

