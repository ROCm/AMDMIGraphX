#ifndef ROCM_GUARD_ROCM_STDDEF_HPP
#define ROCM_GUARD_ROCM_STDDEF_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

using nullptr_t = decltype(nullptr);
using size_t    = uint64_t;
using ptrdiff_t = int64_t;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_STDDEF_HPP
