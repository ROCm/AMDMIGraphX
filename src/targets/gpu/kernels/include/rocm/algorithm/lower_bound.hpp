#ifndef ROCM_GUARD_ROCM_ALGORITHM_LOWER_BOUND_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_LOWER_BOUND_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/upper_bound.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class T, class Compare>
constexpr Iterator lower_bound(Iterator first, Iterator last, const T& value, Compare comp)
{
    return upper_bound(first, last, value, [&](auto&& a, auto&& b) { return not comp(b, a); });
}

template <class Iterator, class T>
constexpr Iterator lower_bound(Iterator first, Iterator last, const T& value)
{
    return lower_bound(first, last, value, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_LOWER_BOUND_HPP
