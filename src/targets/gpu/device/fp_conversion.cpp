#include <migraphx/gpu/device/fp_conversion.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
fp_conversion(hipStream_t stream, const shape& output_shape, const std::vector<argument>& args)
{
    args.back().visit([&](auto output) {
        args.front().visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr      = device_cast(output.data());
            gs_launch(stream,
                      output_shape.elements())([=](auto i) { output_ptr[i] = input_ptr[i]; });
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
