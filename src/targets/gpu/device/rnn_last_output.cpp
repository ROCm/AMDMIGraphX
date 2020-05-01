#include <migraphx/gpu/device/rnn_last_output.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_last_output(hipStream_t stream,
                     const argument& result,
                     const argument& arg_hs,
                     const argument& arg_sl,
                     bool is_reverse)
{
    auto input_shape   = arg_hs.get_shape();
    auto out_comp_lens = input_shape.lens();
    out_comp_lens[0]   = 1;
    shape out_comp_shape{input_shape.type(), out_comp_lens};

    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            auto in_s           = make_hip_shape<4>(input_shape);
            auto out_s          = make_hip_shape<4>(out_comp_shape);
            gs_launch(stream, result.get_shape().elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto d   = idx[1];
                auto b   = idx[2];
                auto l   = sl_data[b];
                if(is_reverse or d == 1)
                {
                    idx[0] = 0;
                }
                else
                {
                    idx[0] = l - 1;
                }
                out_data[i] = in_data[in_s.index(idx)];
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
