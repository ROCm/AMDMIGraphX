#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/scatter.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class Shape>
constexpr auto get_rank(const Shape&)
{
    return decltype(typename Shape::hip_index{}.size()){};
}

argument scatter(
    hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2, int64_t axis)
{
    auto ds            = arg0.get_shape();
    auto inds          = arg1.get_shape();
    auto upds          = arg2.get_shape();
    auto axis_dim_size = ds.lens()[axis];
    hip_visit_all(result, arg0)([&](auto output, auto data) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        gs_launch(stream, ds.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        hip_visit_all(arg1, arg2)([&](auto indices, auto update) {
            const auto* indices_ptr = device_cast(indices.data());
            if constexpr(get_rank(update.get_shape()) == get_rank(output.get_shape()))
            {
                gs_launch(stream, inds.elements())([=](auto i) __device__ {
                    auto out_idx    = indices.get_shape().multi(i);
                    auto upd_idx    = out_idx;
                    auto idx        = indices_ptr[i];
                    idx             = idx < 0 ? idx + axis_dim_size : idx;
                    out_idx[axis]   = idx;
                    output[out_idx] = update[upd_idx];
                });
            }
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
