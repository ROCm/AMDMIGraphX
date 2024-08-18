#ifndef MIGRAPHX_GUARD_MIGRAPHX_COPY_ASSIGNABLE_FUNCTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COPY_ASSIGNABLE_FUNCTION_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template<class F>
struct copy_assignable_function_wrapper
{
    optional<F> f;

    copy_assignable_function_wrapper(F pf) : f(std::move(pf))
    {}
    copy_assignable_function_wrapper(const copy_assignable_function_wrapper& other) = default;
    copy_assignable_function_wrapper(copy_assignable_function_wrapper&& other) = default;
    copy_assignable_function_wrapper& operator=(copy_assignable_function_wrapper other)
    {
        f.swap(other.f);
        return *this;
    }

    template<class... Ts>
    auto operator()(Ts&&... xs) const -> decltype((*f)(std::forward<Ts>(xs)...))
    {
        return (*f)(std::forward<Ts>(xs)...);
    }
};

template<class F>
using copy_assignable_function = std::conditional_t<std::is_copy_assignable<F>{}, F, copy_assignable_function_wrapper<F>>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COPY_ASSIGNABLE_FUNCTION_HPP
