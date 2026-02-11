#ifndef ROCM_GUARD_UTILITY_SWAP_HPP
#define ROCM_GUARD_UTILITY_SWAP_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T>
constexpr void swap(T& a, T& b) noexcept
{
    T tmp = static_cast<T&&>(a);
    a     = static_cast<T&&>(b);
    b     = static_cast<T&&>(tmp);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_UTILITY_SWAP_HPP
