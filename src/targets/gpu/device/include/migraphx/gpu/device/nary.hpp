#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_NARY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_NARY_HPP

#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/array.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// template <class T>
// using vec4 = T __attribute__((ext_vector_type(4)));

template <class T, std::size_t N>
using vec = T __attribute__((ext_vector_type(N)));

template <std::size_t N, class T>
__device__ __host__ vec<T, N>* as_vec(T* x)
{
    return reinterpret_cast<vec<T, N>*>(x);
}

template <std::size_t N, class T>
__device__ __host__ T* as_pointer(vec<T, N>* x)
{
    return reinterpret_cast<T*>(x);
}

template <std::size_t N, class... Ts>
auto pack_vec(Ts... xs)
{
    return [=](auto f, std::size_t n) { return f(as_vec<4>(xs)[n]...); };
}

template <class F, class... Arguments>
auto nary_nonstandard_impl(hipStream_t stream, F f, argument result, Arguments... args)
{
    std::size_t nelements = result.get_shape().elements();
    hip_visit_all(result, args...)([&](auto output, auto... inputs) {
        gs_launch(stream, nelements)([=](auto i) {
            auto idx  = output.get_shape().multi(i);
            output[i] = f(inputs[idx]...);
        });
    });
}

template <class F>
void trinary_broadcast_vec_impl(hipStream_t stream,
                                F f,
                                const argument& result,
                                const argument& arg1,
                                const argument& arg2,
                                const argument& arg3)
{
    const auto& output_shape = result.get_shape();
    const auto& b_shape      = arg3.get_shape();
    auto bdim =
        std::distance(b_shape.strides().begin(),
                      std::find_if(b_shape.strides().begin(), b_shape.strides().end(), [](auto x) {
                          return x != 0;
                      }));
    auto bdim_len         = output_shape.lens()[bdim];
    auto bdim_stride      = output_shape.strides()[bdim];
    auto bdim_next_stride = bdim_stride * bdim_len;

    visit_all(result, arg1, arg2, arg3)([&](auto output, auto input1, auto input2, auto input3) {
        using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;
        auto* xp   = as_vec<4>(device_cast(input1.data()));
        auto* yp   = as_vec<4>(device_cast(input2.data()));
        auto* zp   = as_vec<4>(device_cast(input3.data()));
        auto* outp = as_vec<4>(device_cast(output.data()));

        const std::size_t vec_size     = 4;
        const std::size_t nlocal       = 1024;
        const std::size_t nglobal      = 256 * nlocal;
        const std::size_t n            = output.size() / vec_size;
        const std::size_t bdim_vec_len = bdim_len / vec_size;

        launch(stream, nglobal, nlocal)([=](auto idx) __device__ {
            MIGRAPHX_DEVICE_SHARED vec<type, 4> buffer[2048 / vec_size];
            // Load bias into LDS
            for(size_t i = idx.local; i < bdim_vec_len; i += nlocal)
            {
                buffer[i] = zp[i];
            }
            __syncthreads();
            auto* bp = as_pointer(buffer);
            // Process the data
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                auto bidx        = ((i * vec_size) % bdim_next_stride) / bdim_stride;
                auto b           = bp[bidx];
                vec<type, 4> x   = xp[i];
                vec<type, 4> y   = yp[i];
                vec<type, 4> out = outp[i];
                for(std::size_t j = 0; j < vec_size; j++)
                {
                    out[j] = f(x[j], y[j], b);
                }
                outp[i] = out;
            }
        });
    });
}

template <class F>
void trinary_broadcast_impl(hipStream_t stream,
                            F f,
                            const argument& result,
                            const argument& arg1,
                            const argument& arg2,
                            const argument& arg3)
{
    const auto& output_shape = result.get_shape();
    const auto& b_shape      = arg3.get_shape();
    auto bdim =
        std::distance(b_shape.strides().begin(),
                      std::find_if(b_shape.strides().begin(), b_shape.strides().end(), [](auto x) {
                          return x != 0;
                      }));
    auto bdim_len         = output_shape.lens()[bdim];
    auto bdim_stride      = output_shape.strides()[bdim];
    auto bdim_next_stride = bdim_stride * bdim_len;

    visit_all(result, arg1, arg2, arg3)([&](auto output, auto input1, auto input2, auto input3) {
        using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;
        auto* xp   = device_cast(input1.data());
        auto* yp   = device_cast(input2.data());
        auto* zp   = device_cast(input3.data());
        auto* outp = device_cast(output.data());

        const std::size_t nlocal  = 1024;
        const std::size_t nglobal = 256 * nlocal;
        const std::size_t n       = output.size();

        launch(stream, nglobal, nlocal)([=](auto idx) __device__ {
            MIGRAPHX_DEVICE_SHARED type buffer[2048];
            // Load bias into LDS
            for(size_t i = idx.local; i < bdim_len; i += nlocal)
            {
                buffer[i] = zp[i];
            }
            __syncthreads();
            // Process the data
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                auto bidx = (i % bdim_next_stride) / bdim_stride;
                auto b    = buffer[bidx];
                type x    = xp[i];
                type y    = yp[i];
                outp[i]   = f(x, y, b);
            }
        });
    });
}

template <class F>
void binary_broadcast_vec_impl(
    hipStream_t stream, F f, const argument& result, const argument& arg1, const argument& arg2)
{
    const auto& output_shape = result.get_shape();
    const auto& b_shape      = arg2.get_shape();
    auto bdim =
        std::distance(b_shape.strides().begin(),
                      std::find_if(b_shape.strides().begin(), b_shape.strides().end(), [](auto x) {
                          return x != 0;
                      }));
    auto bdim_len         = output_shape.lens()[bdim];
    auto bdim_stride      = output_shape.strides()[bdim];
    auto bdim_next_stride = bdim_stride * bdim_len;

    visit_all(result, arg1, arg2)([&](auto output, auto input1, auto input2) {
        using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;
        auto* xp   = as_vec<4>(device_cast(input1.data()));
        auto* yp   = as_vec<4>(device_cast(input2.data()));
        auto* outp = as_vec<4>(device_cast(output.data()));

        const std::size_t vec_size     = 4;
        const std::size_t nlocal       = 1024;
        const std::size_t nglobal      = 256 * nlocal;
        const std::size_t n            = output.size() / vec_size;
        const std::size_t bdim_vec_len = bdim_len / vec_size;

        launch(stream, nglobal, nlocal)([=](auto idx) __device__ {
            MIGRAPHX_DEVICE_SHARED vec<type, 4> buffer[2048 / vec_size];
            // Load bias into LDS
            for(size_t i = idx.local; i < bdim_vec_len; i += nlocal)
            {
                buffer[i] = yp[i];
            }
            __syncthreads();
            auto* bp = as_pointer(buffer);
            // Process the data
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                auto bidx        = ((i * vec_size) % bdim_next_stride) / bdim_stride;
                auto b           = bp[bidx];
                vec<type, 4> x   = xp[i];
                vec<type, 4> out = outp[i];
                for(std::size_t j = 0; j < vec_size; j++)
                {
                    out[j] = f(x[j], b);
                }
                outp[i] = out;
            }
        });
    });
}

template <class F>
void binary_broadcast_impl(
    hipStream_t stream, F f, const argument& result, const argument& arg1, const argument& arg2)
{
    const auto& output_shape = result.get_shape();
    const auto& b_shape      = arg2.get_shape();
    auto bdim =
        std::distance(b_shape.strides().begin(),
                      std::find_if(b_shape.strides().begin(), b_shape.strides().end(), [](auto x) {
                          return x != 0;
                      }));
    auto bdim_len         = output_shape.lens()[bdim];
    auto bdim_stride      = output_shape.strides()[bdim];
    auto bdim_next_stride = bdim_stride * bdim_len;

    visit_all(result, arg1, arg2)([&](auto output, auto input1, auto input2) {
        using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;
        auto* xp   = device_cast(input1.data());
        auto* yp   = device_cast(input2.data());
        auto* outp = device_cast(output.data());

        const std::size_t nlocal  = 1024;
        const std::size_t nglobal = 256 * nlocal;
        const std::size_t n       = output.size();

        launch(stream, nglobal, nlocal)([=](auto idx) __device__ {
            MIGRAPHX_DEVICE_SHARED type buffer[2048];
            // Load bias into LDS
            for(size_t i = idx.local; i < bdim_len; i += nlocal)
            {
                buffer[i] = yp[i];
            }
            __syncthreads();
            // Process the data
            for(size_t i = idx.global; i < n; i += nglobal)
            {
                auto bidx = (i % bdim_next_stride) / bdim_stride;
                auto b    = buffer[bidx];
                type x    = xp[i];
                outp[i]   = f(x, b);
            }
        });
    });
}

template <class F, class... Arguments>
void nary_broadcast_impl(
    hipStream_t stream, F f, argument result, argument barg, Arguments... args)
{
    const auto& output_shape = result.get_shape();
    const auto& b_shape      = barg.get_shape();
    auto bdim =
        std::distance(b_shape.strides().begin(),
                      std::find_if(b_shape.strides().begin(), b_shape.strides().end(), [](auto x) {
                          return x != 0;
                      }));
    auto bdim_len         = output_shape.lens()[bdim];
    auto bdim_stride      = output_shape.strides()[bdim];
    auto bdim_next_stride = bdim_stride * bdim_len;

    const std::size_t nlocal  = 1024;
    const std::size_t nglobal = 256 * nlocal;
    std::size_t nelements = result.get_shape().elements();
    hip_visit_all(result, barg, args...)([&](auto output, auto binput, auto... inputs) {
        using type = typename decltype(output)::value_type;
        launch(stream, nglobal, nlocal)([=](auto idx) __device__ {
            MIGRAPHX_DEVICE_SHARED type buffer[2048];
            // Load bias into LDS
            for(size_t i = idx.local; i < bdim_len; i += nlocal)
            {
                buffer[i] = binput.data()[i];
            }
            __syncthreads();
            // Process the data
            for(size_t i = idx.global; i < nelements; i += nglobal)
            {
                auto bidx = (i % bdim_next_stride) / bdim_stride;
                auto b    = buffer[bidx];
                output.data()[i]   = f(inputs.data()[i]..., b);
            }
        });
    });
}

template <class F, class... Arguments>
void nary_standard_vec_impl(hipStream_t stream, F f, argument result, Arguments... args)
{
    // assert(x.get_shape().elements() == y.get_shape().elements());
    const auto& output_shape = result.get_shape();
    visit_all(result, args...)([&](auto output, auto... inputs) {
        using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;
        const std::size_t vec_size = 4;
        auto data                  = pack_vec<4>(device_cast(inputs.data())...);
        auto* outp                 = as_vec<4>(device_cast(output.data()));
        gs_launch(stream, output_shape.elements() / vec_size)([=](auto i) {
            vec<type, 4> out = outp[i];
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
    });
}

template <class F, class... Arguments>
void nary_standard_impl(hipStream_t stream, F f, argument result, Arguments... args)
{
    std::size_t nelements = result.get_shape().elements();
    hip_visit_all(result, args...)([&](auto output, auto... inputs) {
        gs_launch(stream, nelements)([=](auto i) { output.data()[i] = f(inputs.data()[i]...); });
    });
}

template <class F, class... Arguments>
void nary_impl(hipStream_t stream, F f, argument result, Arguments... args)
{
    bool standard = all_of({args.get_shape()...}, [](const shape& s) { return s.standard(); });
    bool packed   = all_of({args.get_shape()...}, [](const shape& s) { return s.packed(); });
    bool same_shapes =
        all_of({args.get_shape()...}, [&](const shape& s) { return s == result.get_shape(); });
    if(standard or (packed and same_shapes))
        nary_standard_impl(stream, f, result, args...);
    else
        nary_nonstandard_impl(stream, f, result, args...);
}

template <class... Arguments>
auto nary_nonstandard(hipStream_t stream, argument result, Arguments... args)
{
    return [=](auto f) { nary_nonstandard_impl(stream, f, result, args...); };
}

template <class... Arguments>
auto nary_standard(hipStream_t stream, argument result, Arguments... args)
{
    return [=](auto f) { nary_standard_impl(stream, f, result, args...); };
}

template <class... Arguments>
auto nary(hipStream_t stream, argument result)
{
    return [=](auto f) { nary_standard_impl(stream, f, result); };
}

template <class... Arguments>
auto
nary(hipStream_t stream, argument result, Arguments... args)
{

    return [=](auto f) {
        auto barg = back_args(args...);
        pop_back_args(args...)([&](auto&&... args2) {
            auto bshape = barg.get_shape();
            const bool standard = all_of({args2.get_shape()...}, [](const shape& s) { return s.standard(); });
            const bool same_shapes =
                all_of({args2.get_shape()...}, [&](const shape& s) { return s == result.get_shape(); });
            // TODO: Check result and args shape is the same
            if(standard and same_shapes and bshape.broadcasted() and
               not bshape.scalar())
            {
                auto not_zero       = [](auto x) { return x != 0; };
                const auto& strides = bshape.strides();
                auto b_it           = std::find_if(strides.begin(), strides.end(), not_zero);
                auto b_idx          = std::distance(strides.begin(), b_it);
                auto b_len          = result.get_shape().lens()[b_idx];
                auto b_stride       = result.get_shape().strides()[b_idx];
                assert(bshape.lens()[b_idx] == b_len);
                if(b_len <= 2048 and std::none_of(std::next(b_it), strides.end(), not_zero))
                {
                    nary_broadcast_impl(stream, f, result, barg, args2...);

                    // const bool divisible_by_4 = (b_len % 4 == 0) and (b_stride % 4 == 0) and
                    //                             (arg1.get_shape().elements() % 4 == 0);
                    // if(divisible_by_4)
                    //     binary_broadcast_vec_impl(stream, f, result, arg1, arg);
                    // else
                    //     binary_broadcast_impl(stream, f, result, arg1, arg);
                    // return;
                }
            }
        });
        nary_impl(stream, f, result, args...);
    };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
