
#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_UNARY_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_UNARY_HPP

#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>

namespace migraph {
namespace gpu {
namespace device {

template<class F>
void unary(argument x, argument result, F f)
{
    if(x.get_shape().standard())
        unary_standard(x, result, f);
    else
        unary_nonstandard(x, result, f);
}

template<class F>
void unary_nonstandard(argument x, argument result, F f)
{
    auto output_shape = result.get_shape();
    auto input_shape = x.get_shape();
    visit_all(result, x)([&](auto output, auto input) {
        visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
            hip_tensor_descriptor<ndim> x_desc(input_shape.lens(), input_shape.strides());
            hip_tensor_descriptor<ndim> out_desc(output_shape.lens(), output_shape.strides());
            auto* xp  = input.data();
            auto* outp = output.data();
            gs_launch(input_shape.elements())([=](auto i) {
                size_t xidx = x_desc.linear(out_desc.multi(i));
                outp[i]       = f(xp[xidx]);
            });
        });
    });
}

template<class F>
void unary_standard(argument x, argument result, F f)
{
    auto output_shape = result.get_shape();
    auto input_shape = x.get_shape();
    visit_all(result, x)([&](auto output, auto input) {
        auto* xp  = input.data();
        auto* outp = output.data();
        gs_launch(input_shape.elements())([=](auto i) {
            outp[i]       = f(xp[i]);
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
