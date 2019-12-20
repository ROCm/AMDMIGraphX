#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ARG_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ARG_OP_HPP

#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
struct val_index
{
    T val;
    int64_t index;
};

template <class T>
MIGRAPHX_DEVICE_CONSTEXPR val_index<T> make_val_index(T v)
{
    return {v, -1};
}

template <class T>
MIGRAPHX_DEVICE_CONSTEXPR val_index<T> make_val_index(T v, int64_t i)
{
    return {v, i};
}

struct argmax_op
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR val_index<T> operator()(val_index<T> x, val_index<T> y) const
    {
        if(x.val > y.val)
            return x;
        else if(x.val < y.val)
            return y;
        else
        {
            return (x.index < y.index) ? x : y;
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR auto init() const { return lowest(); }
};

struct argmin_op
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR val_index<T> operator()(val_index<T> x, val_index<T> y) const
    {
        if(x.val < y.val)
            return x;
        else if(x.val > y.val)
            return y;
        else
        {
            return (x.index < y.index) ? x : y;
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR auto init() const { return highest(); }
};

template <class Op>
void arg_op(Op op, hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    auto arg_shape        = arg.get_shape();
    auto batch_lens       = arg_shape.lens();
    size_t batch_item_num = batch_lens[axis];
    batch_lens[axis]      = 1;
    migraphx::shape batch_shape{arg_shape.type(), batch_lens};

    hip_visit_all(arg, arg_shape, batch_shape)([&](auto input, auto arg_s, auto batch_s) {
        auto output = device_cast(result.get<int64_t>().data());
        using type  = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
        // use one block for items in one batch.
        const size_t max_block_size  = 256;
        const std::size_t block_size = compute_block_size(batch_item_num, max_block_size);
        gs_launch(stream,
                  batch_shape.elements() * block_size,
                  block_size)([=](auto i, auto idx) __device__ {
            auto batch_idx = batch_s.multi(i / block_size);
            auto data_idx  = batch_idx;
            auto init      = make_val_index<type>(op.init());

            auto op_output =
                block_reduce<max_block_size>(idx, op, init, batch_item_num, [&](auto j) __device__ {
                    data_idx[axis] = j;
                    return make_val_index(input[arg_s.index(data_idx)], j);
                });

            if(idx.local == 0)
            {
                output[batch_s.index(batch_idx)] = op_output.index;
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
