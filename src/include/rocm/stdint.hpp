#ifndef ROCM_GUARD_ROCM_STDINT_HPP
#define ROCM_GUARD_ROCM_STDINT_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

#ifdef __HIPCC_RTC__
using int8_t   = __hip_int8_t;
using uint8_t  = __hip_uint8_t;
using int16_t  = __hip_int16_t;
using uint16_t = __hip_uint16_t;
using int32_t  = __hip_int32_t;
using uint32_t = __hip_uint32_t;
using int64_t  = __hip_int64_t;
using uint64_t = __hip_uint64_t;
#else
using int8_t   = std::int8_t;
using uint8_t  = std::uint8_t;
using int16_t  = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t  = std::int32_t;
using uint32_t = std::uint32_t;
using int64_t  = std::int64_t;
using uint64_t = std::uint64_t;
#endif

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_STDINT_HPP
