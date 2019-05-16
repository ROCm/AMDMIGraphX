#ifndef MIGRAPHX_GUARD_RTGLIB_MAKE_SIGNED_HPP
#define MIGRAPHX_GUARD_RTGLIB_MAKE_SIGNED_HPP

#include <migraphx/config.hpp>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
typename std::conditional_t<std::is_integral<T>{}, std::make_signed<T>, std::enable_if<true, T>>::
    type
    make_signed(T x)
{
    return x;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
