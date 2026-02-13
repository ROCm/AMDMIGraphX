#ifndef ROCM_GUARD_UTILITY_MOVE_HPP
#define ROCM_GUARD_UTILITY_MOVE_HPP

#include <rocm/config.hpp>
#include <rocm/type_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T>
constexpr remove_reference_t<T>&& move(T&& x) noexcept
{
    return static_cast<remove_reference_t<T>&&>(x);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_UTILITY_MOVE_HPP
