#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP

#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>
#include <migraph/functional.hpp>
#include <migraph/ranges.hpp>

namespace migraph {
namespace gpu {
namespace device {

template <class F, class... Arguments>
auto nary_nonstandard_impl(F f, argument result, Arguments... args)
{
    const auto& output_shape = result.get_shape();
    visit_all(result, args...)([&](auto output, auto... inputs) {
        visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
            auto data = pack(
                std::make_pair(hip_tensor_descriptor<ndim>{inputs.get_shape()}, inputs.data())...);
            hip_tensor_descriptor<ndim> out_desc(output_shape);
            auto* outp = output.data();
            gs_launch(output_shape.elements())([=](auto i) {
                data([&](auto&&... ps) {
                    auto outidx = out_desc.multi(i);
                    outp[i]     = f(ps.second[ps.first.linear(outidx)]...);
                });
            });
        });
    });
}

template <class... Arguments>
auto nary_nonstandard(argument result, Arguments... args)
{
    return [=](auto f) { return nary_nonstandard_impl(f, result, args...); };
}

inline auto binary_broadcast(argument result, argument arg1, argument arg2)
{
    return [=](auto f) {
        const auto& b_shape = arg2.get_shape();
        auto bdim           = std::distance(b_shape.strides().begin(),
                                  std::find_if(b_shape.strides().begin(),
                                               b_shape.strides().end(),
                                               [](auto x) { return x != 0; }));
        auto bdim_len       = b_shape.lens()[bdim];

        visit_all(result, arg1, arg2)([&](auto output, auto input1, auto input2) {
            using type = std::remove_cv_t<typename decltype(output)::value_type>;
            auto* xp   = input1.data();
            auto* yp   = input2.data();
            auto* outp = output.data();

            const std::size_t nlocal  = 256;
            const std::size_t nglobal = 256 * nlocal;
            const std::size_t n       = output.size();

            launch(nglobal, nlocal)([=](auto idx) __device__ {
                __shared__ type buffer[2048];
                for(size_t i = idx.local; i < bdim_len; i += nlocal)
                {
                    buffer[i] = yp[i];
                }
                __syncthreads();
                for(size_t i = idx.local; i < bdim_len; i += nlocal)
                {
                    auto b = buffer[i];
                    for(size_t j = idx.global; j < n; j += nglobal)
                    {
                        outp[j] = f(xp[j], b);
                    }
                }
            });
        });
    };
}

template <class... Arguments>
auto nary_standard(argument result, Arguments... args)
{
    return [=](auto f) {
        // assert(x.get_shape().elements() == y.get_shape().elements());
        const auto& output_shape = result.get_shape();
        visit_all(result, args...)([&](auto output, auto... inputs) {
            auto data  = pack(inputs.data()...);
            auto* outp = output.data();
            gs_launch(output_shape.elements())(
                [=](auto i) { data([&](auto... xps) { outp[i] = f(xps[i]...); }); });
        });
    };
}

template <class... Arguments>
auto nary_impl(argument result, Arguments... args)
{
    return [=](auto f) {
        bool standard = all_of({args.get_shape()...}, [](const shape& s) { return s.standard(); });
        bool packed   = all_of({args.get_shape()...}, [](const shape& s) { return s.packed(); });
        bool same_shapes =
            all_of({args.get_shape()...}, [&](const shape& s) { return s == result.get_shape(); });
        if(standard or (packed and same_shapes))
            nary_standard(result, args...)(f);
        else
            nary_nonstandard(result, args...)(f);

    };
}

template <class... Arguments>
auto nary(argument result, Arguments... args)
{
    return nary_impl(result, args...);
}

inline auto nary(argument result, argument arg1, argument arg2)
{
    return [=](auto f) {
        // TODO: Check for one broadcast stride
        // TODO: Check result and arg1 shape is the same
        // TODO: CHeck that broadcast shape doesnt have more than 2048 elements
        if(arg1.get_shape().standard() and arg2.get_shape().broadcasted() and
           std::count_if(arg2.get_shape().strides().begin(),
                         arg2.get_shape().strides().end(),
                         [](auto x) { return x != 0; }) == 1)
        {
            binary_broadcast(result, arg1, arg2)(f);
        }
        else
        {
            nary_impl(result, arg1, arg2)(f);
        }
    };
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
