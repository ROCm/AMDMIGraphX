#ifndef ROCM_GUARD_ROCM_ALGORITHM_FILL_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_FILL_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class T>
constexpr void fill(Iterator first, Iterator last, const T& value)
{
    for(; first != last; ++first)
        *first = value;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_FILL_HPP
