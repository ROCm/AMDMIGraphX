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

    // helper function matches "pow" operator and 3.0
    auto pow_fn() const { return f("pow")(used_once(), arg(0).bind("x"), arg(1)(has_value(3.0f))); }

    // clang-format off
    // helper function matches the subexpression of the gelu formula:
    // tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x, 3)))
    // clang-format on
    // This is the expression that is actually simplified to the erf function
    // and can be used as a matcher predicate by itself.
    // Also binds the name "x" to the first appearance of the x argument.
    // Note that there is no checking that the "x" here matches the two other appearances
    // of x in the gelu formula.
    auto tanh_sub_fn() const
    {
        // magic number * x**3
        auto const_times_pow_fn = f("mul")(either_arg(0, 1)(has_value(0.044715f), pow_fn()));
        // add x to result of const_times_pow_fn
        // The instruction matched by any() here should be the same x that was bound in pow_fn(),
        // but matcher has no way to ensure they're the same.  Do not bind to label "x" a second
        // time as it leads to undefined results.
        auto add_x_fn = f("add")(either_arg(0, 1)(any(), const_times_pow_fn));

        // multiply by sqrt(2 / pi)
        auto mul_sqrt_2_over_pi =
            f("mul")(either_arg(0, 1)(add_x_fn, has_value(sqrt(M_2_PI), 1e-3)));

        // tanh
        auto tanh_fn = f("tanh")(used_once(), arg(0)(mul_sqrt_2_over_pi));
        return tanh_fn;
    }

    auto matcher() const
    {
        // clang-format off
        // Formula used in gelu.cpp:   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))
        //      but we only match the subexpression without the 0.5 factor:
        //      x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))
        //      because that's how it appears in common bert models.
        //
        // matcher grammar:
        // used_once() means only match an instruction whose output is used only once (otherwise, substituting 
        //   it may not improve performance).
        // bind("x") makes an instruction "x" findable by substitution code such as x_ins = r.instructions["x"]; 
        //   This code assumes that the multiple bindings of x refer to the same value; not checked.
        // any() matches anything; the only purpose is to get a handle for
        //   another instruction such as bind() 
        // any_arg() finds a match in any argument position
        // either_arg() must match both arguments but in either order is OK
        // name() and name_contains() match if name matches
        // clang-format on

        // Each of these calls returns a matcher predicate function which we compose
        // with each other.  They could alternatively have been declared as helper
        // member functions of this struct.
        //
        // Each function matches a step in the algebraic gelu formula.

        // the subfunction returned by tanh_sub_fn() matches the inner part of the formula, then
        // add 1
        auto add_one_fn = f("add")(either_arg(0, 1)(tanh_sub_fn(), has_value(1.0F, 1e-3)));

                    // latest: we tried removing factor of .5 from expression to be matched
                    // auto mul_x_fn = f("mul")(used_once(), either_arg(0, 1)(add_one_fn, any()));
                    // return mul_x_fn;
        // multiply by 0.5
        auto mul_point_5_fn = f("mul")(either_arg(0, 1)(add_one_fn, has_value(0.5F, 1e-3)));

        // multiply by x.  The last two steps have to appear in this order to match the instructions
        // used in real-world bert models.
        auto mul_x_fn = f("mul")(either_arg(0, 1)(mul_point_5_fn, any()));
// auto test_fn=f("mul")(either_arg(0, 1)(mul_point_5_fn, has_value(1.0F, 1e-3)));;
// return test_fn;
        return mul_x_fn;        
    }
};
} // namespace detail

template <class F>
auto gelu_tanh(F f)
{
    return detail::gelu_tanh_matcher<F>{f}.matcher();
}
template <class F>
auto gelu_tanh_inner(F f)
{
    return detail::gelu_tanh_matcher<F>{f}.matcher_inner();
}

inline auto gelu_tanh()
{
    return gelu_tanh([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
