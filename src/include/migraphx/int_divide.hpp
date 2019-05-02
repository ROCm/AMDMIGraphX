#ifndef MIGRAPHX_GUARD_RTGLIB_INT_DIVIDE_HPP
#define MIGRAPHX_GUARD_RTGLIB_INT_DIVIDE_HPP

#include <migraphx/config.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class R, class T, class U>
R floor_divide(T x, U y)
{
    return R(std::floor(double(x) / double(y)));
}

template <class R, class T, class U>
R ceil_divide(T x, U y)
{
    return R(std::ceil(double(x) / double(y)));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
