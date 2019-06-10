#include <migraphx/gpu/device/convert.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void convert(hipStream_t stream,
             const argument& result,
             const argument& arg,
             float scale,
             float shift,
             shape::type_t target_type)
{
    result.visit([&](auto output) {
        arg.visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr      = device_cast(output.data());
            if(target_type == shape::int8_type)
            {
                gs_launch(stream, result.get_shape().elements())([=](auto i) {
                    float res     = input_ptr[i] * scale + shift;
                    int factor    = (res >= 0.0f) ? 1 : -1;
                    output_ptr[i] = static_cast<int8_t>(
                        std::min<float>(std::max<float>(-128.0f, res + factor * 0.5), 127.0f));
                });
            }
            else
            {
                gs_launch(stream, result.get_shape().elements())(
                    [=](auto i) { output_ptr[i] = input_ptr[i] * scale + shift; });
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
