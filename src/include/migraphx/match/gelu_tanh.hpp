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
    auto pow_fn() const
    {
        return match::name(f("pow"))(match::used_once(),
                                     match::arg(1)(match::args(match::has_value(3.0f))));
    }

    auto tanh_fn() const
    {
        return match::name(f("tanh"))(
            match::used_once(),
            match::arg(0)(match::name(f("mul"))(match::either_arg(0, 1)(
                match::args(match::has_value(sqrt(M_2_PI), 1e-3)),
                match::name(f("add"))(match::any_arg(0, 1)(match::name(f("mul"))(match::either_arg(
                    0, 1)(match::args(match::has_value(0.044715f)), pow_fn()))))))));
    }

    auto matcher() const
    {
        return match::name(f("mul"))(
            match::used_once(),
            match::either_arg(0, 1)(
                match::any().bind("x"),
                match::name(f("add"))(match::any_arg(0, 1)(match::name(f("mul"))(
                    match::either_arg(0, 1)(match::args(match::has_value(0.5f)), tanh_fn()))))));
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
    return gelu_tanh([](auto x) { return x; });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
