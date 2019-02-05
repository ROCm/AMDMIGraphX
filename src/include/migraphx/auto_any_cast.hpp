#ifndef MIGRAPHX_GUARD_RTGLIB_AUTO_ANY_CAST_HPP
#define MIGRAPHX_GUARD_RTGLIB_AUTO_ANY_CAST_HPP
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Forward declare any_cast
template <class T>
const T& any_cast(const T&);

namespace detail {

template <class U>
void any_cast()
{
}

template <class T>
struct auto_any_caster
{
    T& x;

    template <class U>
    operator U&()
    {
        return any_cast<U>(x);
    }

    operator T&() { return x; }
};

} // namespace detail

template <class T>
detail::auto_any_caster<T> auto_any_cast(T& x)
{
    return {x};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
