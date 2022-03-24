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
    F f; // currently a function that matches name with gpu:: or cpu::prepended see
         // simplify_algebra.cpp

    // helper function matches "pow" operator and 3.0
    auto pow_fn() const { return f("pow")(used_once(), arg(0).bind("x"), arg(1)(has_value(3.0f))); }

    auto matcher() const
    {
        // Formula used in gelu.cpp:   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x,
        // 3)))) used_once() means only match an instruction whose output is used only once
        // bind("x") makes an instruction "x" findable by substitution code such as x_ins =
        // r.instructions["x"]; any() matches anything; the only purpose is to get a handle for
        // another instruction such as bind() any_arg() finds a match in any argument position
        // either_arg() must match both arguments but in either order is OK
        // name() and name_contains() match if name matches

        // magic number * x**3
        auto a1 = f("mul")(either_arg(0, 1)(has_value(0.044715f), pow_fn()));
        // add x to result of a3
        auto a2 = f("add")(either_arg(0, 1)(any().bind("x"), a1));

        // multiply by sqrt(2 / pi)
        auto a3 = f("mul")(either_arg(0, 1)(a2, has_value(sqrt(M_2_PI), 1e-3)));

        // tanh
        auto a4 = f("tanh")(used_once(), arg(0)(a3));

        // add 1
        auto a5 = f("add")(either_arg(0, 1)(a4, has_value(1.0F, 1e-3)));

        // multiply by x
        auto a6 = f("mul")(used_once(), either_arg(0, 1)(a5, any().bind("x")));

        // multiply by 0.5
        auto a7 = f("mul")(either_arg(0, 1)(a6, has_value(0.5F, 1e-3)));
        return a7;
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
