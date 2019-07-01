#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ARG_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ARG_OP_HPP

#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
struct val_index
{
    T val;
    int64_t index;

    //  MIGRAPHX_DEVICE_CONSTEXPR val_index(T v, int64_t idx) : val(v), index(idx) { }
};

template <class T>
struct argmax_op
{
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

    MIGRAPHX_DEVICE_CONSTEXPR T init() const { return lowest(); }
};

template <class T>
struct argmin_op
{
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

    MIGRAPHX_DEVICE_CONSTEXPR T init() const { return highest(); }
};

template <class T, class Op>
void arg_op(Op op, hipStream_t stream, const argument& result, const argument& arg, int axis)
{
    auto arg_shape        = arg.get_shape();
    auto lens             = arg_shape.lens();
    auto batch_lens       = lens;
    size_t batch_item_num = lens[axis];
    batch_lens[axis]      = 1;
    migraphx::shape batch_shape{arg_shape.type(), batch_lens};

    hip_visit_all(arg, arg_shape, batch_shape)([&](auto input, auto arg_s, auto batch_s) {
        auto output = device_cast(result.get<int64_t>().data());
        // use one block for items in one batch.
        const size_t max_block_size  = 256;
        const std::size_t block_size = compute_block_size(batch_item_num, max_block_size);
        gs_launch(stream, batch_shape.elements() * block_size, block_size)(
            [=](auto i, auto idx) __device__ {
                auto batch_idx    = batch_s.multi(i / block_size);
                auto data_idx     = batch_idx;
                T init_val        = op.init();
                val_index<T> init = {init_val, -1};

                auto op_output = block_reduce<max_block_size, Op, val_index<T>>(
                    idx, op, init, batch_item_num, [&](auto j) __device__ {
                        data_idx[axis] = j;
                        T val          = input[arg_s.index(data_idx)];
                        return val_index<T>{val, static_cast<int64_t>(j)};
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
