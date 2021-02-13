#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_LAYERNORM_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_LAYERNORM_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct layernorm_matcher
{
    F f;
    template <class... Ts>
    static auto multibroadcast_op(Ts... xs)
    {
        return match::name("multibroadcast")(match::arg(0)(xs...));
    }

    auto x_minus_mean() const
    {
        return match::name(f("sub"))(
            match::arg(0)(match::any().bind("x")),
            match::arg(1)(multibroadcast_op(match::name(f("reduce_mean")))));
    }

    auto variance() const
    {
        return match::name(f("reduce_mean"))(match::arg(0)(
            match::name(f("pow"))(match::arg(0)(x_minus_mean()),
                                  match::arg(1)(multibroadcast_op(match::has_value(2.0f))))));
    }

    auto layernorm_onnx() const
    {
        return match::name(f("div"))(
            match::arg(0)(x_minus_mean()),

            match::arg(1)(multibroadcast_op(
                match::name(f("sqrt"))(match::arg(0)(match::name(f("add"))(match::either_arg(0, 1)(
                    variance(), multibroadcast_op(match::has_value(1e-12f)))))))));
    }

    auto matcher() const { return layernorm_onnx(); }
};
} // namespace detail

template <class F>
auto layernorm(F f)
{
    return detail::layernorm_matcher<F>{f}.matcher();
}

inline auto layernorm()
{
    return layernorm([](auto x) { return x; });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
