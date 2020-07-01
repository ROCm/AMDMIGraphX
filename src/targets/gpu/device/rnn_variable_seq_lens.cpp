#include <migraphx/gpu/device/rnn_variable_seq_lens.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_var_sl_shift_sequence(hipStream_t stream,
                               const argument& result,
                               const argument& arg_hs,
                               const argument& arg_sl)
{
    auto output_shape = result.get_shape();
    int64_t max_len   = output_shape.lens()[0];
    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        auto out_s          = make_hip_shape<3>(output_shape);
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            gs_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto t   = idx[0];
                auto b   = idx[1];
                auto l   = sl_data[b];
                auto val = in_data[0];
                val      = 0;
                if(t >= max_len - l)
                {
                    auto in_idx = idx;
                    in_idx[0] -= (max_len - l);
                    val = in_data[out_s.index(in_idx)];
                }
                out_data[i] = val;
            });
        });
    });
}

void rnn_var_sl_shift_output(hipStream_t stream,
                             const argument& result,
                             const argument& arg_hs,
                             const argument& arg_sl,
                             bool is_reverse)
{
    auto output_shape = result.get_shape();
    int64_t max_len   = output_shape.lens()[0];
    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        auto out_s          = make_hip_shape<4>(output_shape);
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            gs_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto t   = idx[0];
                auto d   = idx[1];
                auto b   = idx[2];
                auto l   = sl_data[b];
                auto val = in_data[0];
                val      = 0;
                if(t < l)
                {
                    int offset  = (d == 1 or is_reverse) ? 1 : 0;
                    auto in_idx = idx;
                    in_idx[0] += offset * (max_len - l);
                    val = in_data[out_s.index(in_idx)];
                }
                out_data[i] = val;
            });
        });
    });
}

void rnn_var_sl_last_output(hipStream_t stream,
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
