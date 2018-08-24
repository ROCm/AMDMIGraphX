#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_BINARY_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_BINARY_HPP

#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>

namespace migraph {
namespace gpu {
namespace device {

template<class F>
void binary(argument x, argument y, argument result, F f)
{
    if(x.get_shape().standard())
        binary_standard(x, y, result, f);
    else
        binary_nonstandard(x, y, result, f);
}

template<class F>
void binary_nonstandard(argument x, argument y, argument result, F f)
{
    auto output_shape = result.get_shape();
    auto input_shape = x.get_shape();
    visit_all(result, x, y)([&](auto output, auto input1, auto input2) {
        visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
            hip_tensor_descriptor<ndim> x_desc(x.get_shape().lens(), x.get_shape().strides());
            hip_tensor_descriptor<ndim> y_desc(y.get_shape().lens(), y.get_shape().strides());
            hip_tensor_descriptor<ndim> out_desc(output_shape.lens(), output_shape.strides());
            auto* xp  = input1.data();
            auto* yp  = input2.data();
            auto* outp = output.data();
            gs_launch(input_shape.elements())([=](auto i) {
                auto outidx = out_desc.multi(i);
                size_t xidx = x_desc.linear(outidx);
                size_t yidx = y_desc.linear(outidx);
                outp[i]       = f(xp[xidx], yp[yidx]);
            });
        });
    });
}

template<class F>
void binary_standard(argument x, argument y, argument result, F f)
{
    assert(x.get_shape().elements() == y.get_shape().elements());
    auto output_shape = result.get_shape();
    auto input_shape = x.get_shape();
    visit_all(result, x, y)([&](auto output, auto input1, auto input2) {
        auto* xp  = input1.data();
        auto* yp  = input2.data();
        auto* outp = output.data();
        gs_launch(input_shape.elements())([=](auto i) {
            outp[i]       = f(xp[i], yp[i]);
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
