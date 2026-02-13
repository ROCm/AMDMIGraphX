#ifndef ROCM_GUARD_UTILITY_FORWARD_HPP
#define ROCM_GUARD_UTILITY_FORWARD_HPP

#include <rocm/config.hpp>
#include <rocm/type_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T>
constexpr T&& forward(remove_reference_t<T>& x) noexcept
{
    return static_cast<T&&>(x);
}

template <class T>
constexpr T&& forward(remove_reference_t<T>&& x) noexcept
{
    static_assert(not is_lvalue_reference<T>{}, "can not forward an rvalue as an lvalue");
    return static_cast<T&&>(x);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_UTILITY_FORWARD_HPP
