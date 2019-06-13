#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/pad.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
pad(hipStream_t stream, argument result, argument arg1, float value, std::vector<std::int64_t> pads)
{
    std::size_t nelements = arg1.get_shape().elements();
    // if(value == std::numeric_limits<float>::lowest())
    // {
    //     visit_all(result)([&](auto output) {
    //             auto* outptr      = output.data();
    //                 gs_launch(stream, nelements)([=](auto i) {
    //                     outptr[i] = std::numeric_limits<typename decltype(output)::value_type>::lowest();
    //                 });
    //             });
    // }
    
    // else
    // {
    //     visit_all(result)([&](auto output) {
    //             auto* outptr      = output.data();
    //                 gs_launch(stream, nelements)([=](auto i) {
    //                     outptr[i] = static_cast<typename decltype(output)::value_type>(value);
    //                 });
    //             });
    // }

    nary(stream, result)([=] { return value; });
    visit_all(result, arg1)([&](auto output, auto input) {
        visit_tensor_size(result.get_shape().lens().size(), [&](auto ndim) {
            std::size_t offsets[ndim];
            std::copy(pads.begin(), pads.begin() + ndim, offsets);
            auto* outptr      = output.data();
            const auto* inptr = input.data();
            hip_tensor_descriptor<ndim> desc_input(input.get_shape());
            hip_tensor_descriptor<ndim> desc_output(output.get_shape());
            gs_launch(stream, nelements)([=](auto i) {
                auto idx = desc_input.multi(i);
                for(std::size_t j = 0; j < ndim; j++)
                {
                    idx[j] += offsets[j];
                }
                outptr[desc_output.linear(idx)] = inptr[i];
            });
        });
    });
    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
