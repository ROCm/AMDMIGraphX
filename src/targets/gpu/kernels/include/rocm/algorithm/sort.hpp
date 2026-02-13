#ifndef ROCM_GUARD_ROCM_ALGORITHM_SORT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_SORT_HPP

#include <rocm/config.hpp>
#include <rocm/assert.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/iter_swap.hpp>
#include <rocm/algorithm/min_element.hpp>
#include <rocm/algorithm/is_sorted.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr void sort(Iterator first, Iterator last, Compare comp)
{
    if(first == last)
        return;
    for(auto i = first; i != last - 1; ++i)
        iter_swap(i, min_element(i, last, comp));
    ROCM_ASSERT(is_sorted(first, last, comp));
}

template <class Iterator>
constexpr void sort(Iterator first, Iterator last)
{
    sort(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_SORT_HPP
