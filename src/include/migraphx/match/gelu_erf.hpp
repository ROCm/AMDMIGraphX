#ifndef MIGRAPHX_GUARD_MATCH_GELU_ERF_HPP
#define MIGRAPHX_GUARD_MATCH_GELU_ERF_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct gelu_erf_matcher
{
    F f;
    auto erf_fn() const
    {
        return match::name(f("erf"))(
            match::used_once(),
            match::arg(0)(match::used_once(),
                          match::name(f("mul"))(match::either_arg(0, 1)(
                              match::none_of(match::has_value(M_SQRT1_2, 1e-3)).bind("x"),
                              match::has_value(M_SQRT1_2, 1e-3)))));
    }

    auto add_erf() const
    {
        return match::name(f("add"))(
            match::used_once(),
            match::either_arg(0, 1)(erf_fn(), match::args(match::has_value(1.0f))));
    }

    auto one_half() const { return match::args(match::has_value(0.5f)); }

    auto matcher() const
    {
        return match::unordered_tree(f("mul"), one_half(), add_erf(), match::any());
    }
};
} // namespace detail

template <class F>
auto gelu_erf(F f)
{
    return detail::gelu_erf_matcher<F>{f}.matcher();
}

inline auto gelu_erf()
{
    return gelu_erf([](auto x) { return x; });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MATCH_GELU_ERF_HPP