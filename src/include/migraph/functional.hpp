#ifndef MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP
#define MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP

#include <utility>

namespace migraph {

namespace detail {

template<class R, class F>
struct fix_f
{
    F f;

    template<class... Ts>
    R operator()(Ts&&... xs) const
    {
        return f(*this, std::forward<Ts>(xs)...);
    }
};

} // namespace detail

/// Implements a fix-point combinator
template<class R, class F>
detail::fix_f<R, F> fix(F f)
{
    return {f};
}

template<class F>
auto fix(F f)
{
    return fix<void>(f);
}

} // namespace migraph

#endif
