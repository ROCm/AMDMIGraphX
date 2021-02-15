#ifndef MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
#define MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct gelu_tanh_matcher
{
    F f;
    auto pow_fn() const { return f("pow")(used_once(), arg(1)(has_value(3.0f))); }

    auto tanh_fn() const
    {
        return f("tanh")(
            used_once(),
            arg(0)(f("mul")(either_arg(0, 1)(has_value(sqrt(M_2_PI), 1e-3),
                                             f("add")(any_arg(0, 1)(f("mul")(either_arg(0, 1)(
                                                 has_value(0.044715f), pow_fn()))))))));
    }

    auto matcher() const
    {
        return f("mul")(used_once(),
                        either_arg(0, 1)(any().bind("x"),
                                         f("add")(any_arg(0, 1)(f("mul")(
                                             either_arg(0, 1)(has_value(0.5f), tanh_fn()))))));
    }
};
} // namespace detail

template <class F>
auto gelu_tanh(F f)
{
    return detail::gelu_tanh_matcher<F>{f}.matcher();
}

inline auto gelu_tanh()
{
    return gelu_tanh([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
