#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP

#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>
#include <migraph/functional.hpp>
#include <migraph/ranges.hpp>

namespace migraph {
namespace gpu {
namespace device {

template <class... Arguments>
auto nary(argument result, Arguments... args)
{
    return [=](auto f) {
        if(all_of({args...}, [](const shape& s) { return s.standard(); }))
            nary_standard(result, args...)(f);
        else
            nary_nonstandard(result, args...)(f);

    };
}

template <class... Arguments>
auto nary_nonstandard(argument result, Arguments... args)
{
    return [=](auto f) {
        auto output_shape = result.get_shape();
        visit_all(result, args...)([&](auto output, auto... inputs) {
            visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
                auto data = make_sequence(
                    std::make_pair(hip_tensor_descriptor<ndim>{inputs.get_shape().lens(),
                                                               inputs.get_shape().strides()},
                                   inputs.data())...);
                hip_tensor_descriptor<ndim> out_desc(output_shape.lens(), output_shape.strides());
                auto* outp = output.data();
                gs_launch(output_shape.elements())([=](auto i) {
                    data([&](auto... ps) {
                        auto outidx = out_desc.multi(i);
                        outp[i]     = f(ps.second[ps.first.linear(outidx)]...);
                    });
                });
            });
        });
    };
}

template <class... Arguments>
auto nary_standard(argument result, Arguments... args)
{
    return [=](auto f) {
        // assert(x.get_shape().elements() == y.get_shape().elements());
        auto output_shape = result.get_shape();
        visit_all(result, args...)([&](auto output, auto... inputs) {
            auto data  = make_sequence(inputs.data()...);
            auto* outp = output.data();
            gs_launch(output_shape.elements())(
                [=](auto i) { data([&](auto... xps) { outp[i] = f(xps[i]...); }); });
        });
    };
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
