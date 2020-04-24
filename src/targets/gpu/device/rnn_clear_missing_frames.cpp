#include <migraphx/gpu/device/rnn_clear_missing_frames.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_clear_missing_frames(hipStream_t stream,
          const argument& result,
          const argument& arg_hs,
          const argument& arg_sl)
{
    auto output_shape = result.get_shape();
    int64_t max_len = output_shape.lens()[0];
    hip_visit_all(result, arg_s, output_shape)([&](auto output, auto input, auto out_s) {
        const auto* in_data = device_cast(input.data());
        auto* out_data = device_cast(output.data());
        arg_sl.visit([&](auto sl) {
            const auto* sl_ptr = device_cast(sl.data());
            gl_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto t = idx[0];
                auto d = idx[1];
                auto b = idx[2];
                auto l = sl_ptr[b];
                auto val = in_data[0];
                val = 0;
                if (t < l)
                {
                    auto in_idx = idx;
                    in_idx[0] += d * (max_len - l);
                    val = input_data[out_s.index(in_idx)];
                }
                out_data[i] = val;
            });
        })
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
