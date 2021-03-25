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
    auto x_minus_mean() const
    {
        return f("sub")(arg(0)(any().bind("x")), arg(1)(skip_broadcasts(f("reduce_mean"))));
    }

    auto variance() const
    {
        return f("reduce_mean")(arg(0)(f("pow")(arg(0)(x_minus_mean()), arg(1)(has_value(2.0f)))));
    }

    auto layernorm_onnx() const
    {
        return f("div")(arg(0)(x_minus_mean()),

                        arg(1)(skip_broadcasts(f("sqrt")(
                            arg(0)(f("add")(either_arg(0, 1)(variance(), has_value(1e-12f))))))));
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
    return layernorm([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
