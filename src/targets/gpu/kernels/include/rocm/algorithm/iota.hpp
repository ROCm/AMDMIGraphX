#ifndef ROCM_GUARD_ROCM_ALGORITHM_IOTA_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_IOTA_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class T>
constexpr void iota(Iterator first, Iterator last, T value)
{
    for(; first != last; ++first, ++value)
        *first = value;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_IOTA_HPP
