/*=============================================================================
    Copyright (c) 2017 Paul Fultz II
    types.hpp
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_TYPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_DEVICE_TYPES_HPP

#include <migraphx/half.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

using gpu_half = __fp16;

namespace detail {
template <class T>
struct device_type
{
    using type = T;
};

template <>
struct device_type<half>
{
    using type = gpu_half;
};

template <class T>
struct host_type
{
    using type = T;
};

template <>
struct device_type<gpu_half>
{
    using type = half;
};

} // namespace detail

template <class T>
using host_type = typename detail::host_type<T>::type;

template <class T>
using device_type = typename detail::device_type<T>::type;

template <class T>
host_type<T> host_cast(T x)
{
    return reinterpret_cast<host_type<T>>(x);
}

template <class T>
host_type<T>* host_cast(T* x)
{
    return reinterpret_cast<host_type<T>*>(x);
}

template <class T>
device_type<T> device_cast(const T& x)
{
    return reinterpret_cast<const device_type<T>&>(x);
}

template <class T>
device_type<T>* device_cast(T* x)
{
    return reinterpret_cast<device_type<T>*>(x);
}

template <class T>
T to_hip_type(T x)
{
    return x;
}

// Hip doens't support __fp16
inline float to_hip_type(gpu_half x) { return x; }

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
