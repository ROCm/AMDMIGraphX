#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_NARY_HPP

#include <migraph/gpu/device/tensor.hpp>
#include <migraph/gpu/device/launch.hpp>
#include <migraph/functional.hpp>
#include <migraph/ranges.hpp>

namespace migraph {
namespace gpu {
namespace device {

template <class T>
using vec4 = T __attribute__((ext_vector_type(4)));

template <class T>
vec4<T>* as_vec4(T* x)
{
    std::uintptr_t a = reinterpret_cast<std::uintptr_t>(x);
    if(a % 32 != 0)
        throw std::runtime_error("Memory not aligned for vector operations");
    return reinterpret_cast<vec4<T>*>(x);
    // return (vec4<T>*)(x);
}

template <class T>
vec4<T> vec4_load(T* x, size_t i)
{
    vec4<T> result;
    auto n = i * 4;
    result[0] = x[n + 0];
    result[1] = x[n + 1];
    result[2] = x[n + 2];
    result[3] = x[n + 3];
    return result;
}

template <class... Ts>
auto pack_vec4(Ts... xs)
{
    return [=](auto f, std::size_t n) { return f(as_vec4(xs)[n]...); };
}

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
        // const auto& output_shape = result.get_shape();
        const auto& b_shape = arg2.get_shape();
        auto bdim           = std::distance(b_shape.strides().begin(),
                                  std::find_if(b_shape.strides().begin(),
                                               b_shape.strides().end(),
                                               [](auto x) { return x != 0; }));
        auto bdim_len       = b_shape.lens()[bdim];

        visit_all(result, arg1, arg2)([&](auto output, auto input1, auto input2) {
            using type = std::remove_cv_t<typename decltype(output)::value_type>;
#if 1
            auto* xp   = input1.data();
            auto* yp   = input2.data();
            auto* outp = output.data();

            const std::size_t nlocal       = 1024;
            const std::size_t nglobal      = 256 * nlocal;
            const std::size_t n            = output.size();

            launch(nglobal, nlocal)([=](auto idx) __device__ {
                __shared__ type buffer[2048];
                // Load bias into LDS
                for(size_t i = idx.local; i < bdim_len; i += nlocal)
                {
                    buffer[i] = yp[i];
                }
                __syncthreads();
                // Process the data
                for(size_t i = idx.global; i < n; i += nglobal)
                {
                    auto bidx      = i % bdim_len;
                    auto b         = buffer[bidx];
                    type x   = xp[i];
                    outp[i] = f(x, b);
                }
            });
#else
            auto* xp   = as_vec4(input1.data());
            auto* yp   = as_vec4(input2.data());
            auto* outp = as_vec4(output.data());

            const std::size_t vec_size     = 4;
            const std::size_t nlocal       = 1024;
            const std::size_t nglobal      = 256 * nlocal;
            const std::size_t n            = output.size() / vec_size;
            const std::size_t rem          = output.size() % vec_size;
            const std::size_t bdim_vec_len = bdim_len / vec_size;
            const std::size_t bdim_vec_rem = bdim_len % vec_size;

            launch(nglobal, nlocal)([=](auto idx) __device__ {
                __shared__ vec4<type> buffer[2048];
                // Load bias into LDS
                for(size_t i = idx.local; i < bdim_vec_len; i += nlocal)
                {
                    buffer[i] = yp[i];
                }
                // Load remaining bias data
                for(size_t i = idx.local; i < bdim_vec_rem; i += nlocal)
                {
                    buffer[bdim_vec_len][i] = yp[bdim_vec_len][i];
                }
                for(size_t i = idx.local; i < (vec_size-bdim_vec_rem); i += nlocal)
                {
                    buffer[bdim_vec_len][i] = yp[0][i];
                }
                __syncthreads();
                // Process the data
                for(size_t i = idx.global; i < n; i += nglobal)
                {
                    auto bidx      = bdim_vec_len == 0 ? 0 : i % bdim_vec_len;
                    auto b         = buffer[bidx];
                    vec4<type> x   = xp[i];
                    vec4<type> out = outp[i];
                    for(std::size_t j = 0; j < vec_size; j++)
                    {
                        out[j] = f(x[j], b[j]);
                    }
                    outp[i] = out;
                }
                for(size_t i = idx.global; i < rem; i += nglobal)
                {
                    outp[n][i] = f(xp[n][i], buffer[bdim_vec_len][i]);
                }
            });
#endif
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
#if 1
            auto data  = pack(inputs.data()...);
            auto* outp = output.data();
            gs_launch(output_shape.elements())(
                [=](auto i) { data([&](auto... xps) { outp[i] = f(xps[i]...); }); });
#else
            using type                 = std::remove_cv_t<typename decltype(output)::value_type>;
            const std::size_t vec_size = 4;
            auto data                  = pack_vec4(inputs.data()...);
            auto* outp                 = as_vec4(output.data());
            gs_launch(output_shape.elements() / vec_size)([=](auto i) {
                vec4<type> out = outp[i];
                data(
                    [&](auto... xs) {
                        for(std::size_t j = 0; j < vec_size; j++)
                        {
                            out[j] = f(xs[j]...);
                        }
                    },
                    i);
                outp[i] = out;
            });
#endif
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
        // TODO: Check result and arg1 shape is the same
        if(arg1.get_shape().standard() and arg2.get_shape().broadcasted() and
           std::count_if(arg2.get_shape().strides().begin(),
                         arg2.get_shape().strides().end(),
                         [](auto x) { return x != 0; }) == 1)
        {
            auto not_zero = [](auto x) { return x != 0; };
            const auto& strides = arg2.get_shape().strides();
            auto stride_it = std::find_if(strides.begin(),
                         strides.end(), not_zero);
            auto stride_idx = std::distance(strides.begin(), stride_it);
            auto stride_len = arg2.get_shape().lens()[stride_idx];
            // TODO: Dont require disibility by 4
            bool divisible_by_4 = (stride_len % 4 == 0) and (arg1.get_shape().elements() % 4 == 0);
            if(divisible_by_4 and stride_len <= 2048 and std::none_of(std::next(stride_it), strides.end(), not_zero)) {
                binary_broadcast(result, arg1, arg2)(f);
                return;
            }
        }
        nary_impl(result, arg1, arg2)(f);
    };
}

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
