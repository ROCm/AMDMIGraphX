#ifndef MIGRAPHX_GUARD_RTGLIB_CLAMP_HPP
#define MIGRAPHX_GUARD_RTGLIB_CLAMP_HPP

#include <migraphx/config.hpp>
#include <migraphx/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template<class U, class T>
U pad_clamp(T x)
{
    if (float_equal(x, std::numeric_limits<T>::lowest()))
        return std::numeric_limits<U>::lowest();
    if (float_equal(x, std::numeric_limits<T>::max()))
        return std::numeric_limits<U>::max();
    return (x < std::numeric_limits<U>::lowest()) ? std::numeric_limits<U>::lowest() : (std::numeric_limits<U>::max() < x) ? std::numeric_limits<U>::max() : U(x);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
