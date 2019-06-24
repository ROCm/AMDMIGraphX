#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/pad.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
pad(hipStream_t stream, argument result, argument arg1, float value, std::vector<std::int64_t> pads)
{
    std::size_t nelements = arg1.get_shape().elements();
    hip_visit_all(result, arg1)([&](auto output, auto input) {
        using type                   = typename decltype(output)::value_type;
        using hip_index             = typename decltype(output)::hip_index;
        type device_val = value;
        if(float_equal(value, std::numeric_limits<float>::lowest()))
        {
            device_val = device_cast(std::numeric_limits<type>::lowest());
        }
        gs_launch(stream, result.get_shape().elements())([=](auto i) { output.data()[i] = device_val; });

        hip_index offsets;
        std::copy(pads.begin(), pads.begin() + offsets.size(), offsets.begin());
        gs_launch(stream, nelements)([=](auto i) {
            auto idx = input.get_shape().multi(i);
            for(std::size_t j = 0; j < offsets.size(); j++)
            {
                idx[j] += offsets[j];
            }
            output[idx] = input.data()[i];
        });        
    });
    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
