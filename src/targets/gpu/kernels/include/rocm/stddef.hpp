#ifndef ROCM_GUARD_ROCM_STDDEF_HPP
#define ROCM_GUARD_ROCM_STDDEF_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

using nullptr_t = decltype(nullptr);

} // namespace ROCM_INLINE_NS
} // namespace rocm

using nullptr_t = rocm::nullptr_t;

#endif // ROCM_GUARD_ROCM_STDDEF_HPP
