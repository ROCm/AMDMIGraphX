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
    F f; // currently a function that matches name with gpu:: or cpu::prepended see simplify_algebra.cpp
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
        // debug only

        // used_once() means only match an instruction whose output is used only once
        // bind("x") makes an instruction "x" findable by substitution code such as x_ins = r.instructions["x"];
        // any() matches anything; the only purpose is to get a handle for another instruction such as bind()
        // any_arg() finds a match in any argument position
        // either_arg() must match both arguments but in either order is OK
        // name() and name_contains() match if name matches

        return f("mul")(used_once(), either_arg(0, 1)(any().bind("x"), has_value(0.7f)));
        // todo:  this isn't matching the gelu function
        // return f("mul")(used_once(),
        //                 either_arg(0, 1)(any().bind("x"),
        //                                  f("add")(any_arg(0, 1)(f("mul")(
        //                                      either_arg(0, 1)(has_value(0.5f), tanh_fn()))))));
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
