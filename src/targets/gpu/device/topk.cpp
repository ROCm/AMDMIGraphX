#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/topk.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument topk(hipStream_t stream, argument val_res, argument ind_res, argument arg, int64_t k, int64_t axis, bool largest)
{
    auto in_s = arg.get_shape();
    auto in_lens = in_s.lens();
    auto out_s = val_res.get_shape();
    auto axis_dim  = in_s.lens()[axis];
    auto comp_lens = in_lens;
    comp_lens[axis] = 1;
    shape comp_s{in_s.type(), comp_lens};
    std::size_t elem_num = comp_s.elements();

    hip_visit_all(val_res, arg, out_s, in_s, comp_s)([&](auto out_val, auto input, auto oss, auto iss, auto css) {
        const auto* data = device_cast(input.data());
        auto* out = device_cast(out_val.data());
        ind_res.visit([&](auto out_ind) {
            auto* ind = device_cast(out_ind.data());
            gs_launch(stream, elem_num, 256)([=](auto i) __device__ {
                auto idx      = css.multi(i);
                // auto in_index = indices_ptr[idx[axis]];
                // in_index      = (in_index < 0) ? in_index + axis_dim_size : in_index;
                // idx[axis]     = in_index;
                // output_ptr[i] = input[idx];
            });
        });
    });

    return argument({val_res, ind_res});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
