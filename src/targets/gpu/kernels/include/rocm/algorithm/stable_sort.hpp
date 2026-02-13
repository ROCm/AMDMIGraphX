#ifndef ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP

#include <rocm/config.hpp>
#include <rocm/assert.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/rotate.hpp>
#include <rocm/algorithm/upper_bound.hpp>
#include <rocm/algorithm/is_sorted.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr void stable_sort(Iterator first, Iterator last, Compare comp)
{
    if(first == last)
        return;
    for(auto i = first; i != last; ++i)
        rotate(upper_bound(first, i, *i, comp), i, i + 1);
    ROCM_ASSERT(is_sorted(first, last, comp));
}

template <class Iterator>
constexpr void stable_sort(Iterator first, Iterator last)
{
    stable_sort(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_STABLE_SORT_HPP
