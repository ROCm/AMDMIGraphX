#ifndef ROCM_GUARD_ROCM_ALGORITHM_MAX_ELEMENT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_MAX_ELEMENT_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/min_element.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr Iterator max_element(Iterator first, Iterator last, Compare comp)
{
    return min_element(first, last, [&](auto&& a, auto&& b) { return comp(b, a); });
}

template <class Iterator>
constexpr Iterator max_element(Iterator first, Iterator last)
{
    return max_element(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_MAX_ELEMENT_HPP
