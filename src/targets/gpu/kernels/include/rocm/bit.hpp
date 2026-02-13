#ifndef ROCM_GUARD_ROCM_BIT_HPP
#define ROCM_GUARD_ROCM_BIT_HPP

#include <rocm/config.hpp>
#include <rocm/type_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <typename To,
          typename From,
          ROCM_REQUIRES(rocm::is_trivially_copyable<To>{} and
                        rocm::is_trivially_copyable<From>{} and sizeof(To) == sizeof(From))>
constexpr To bit_cast(From fr) noexcept
{
    return __builtin_bit_cast(To, fr);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_BIT_HPP
