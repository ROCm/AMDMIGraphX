#ifndef ROCM_GUARD_ROCM_DECLVAL_HPP
#define ROCM_GUARD_ROCM_DECLVAL_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T, class U = T&&>
U private_declval(int);

template <class T>
T private_declval(long);

template <class T>
auto declval() noexcept -> decltype(private_declval<T>(0));

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_DECLVAL_HPP
