#ifndef ROCM_GUARD_ROCM_ALGORITHM_MIN_ELEMENT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_MIN_ELEMENT_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr Iterator min_element(Iterator first, Iterator last, Compare comp)
{
    if(first == last)
        return last;

    Iterator smallest = first;

    while(++first != last)
        if(comp(*first, *smallest))
            smallest = first;

    return smallest;
}

template <class Iterator>
constexpr Iterator min_element(Iterator first, Iterator last)
{
    return min_element(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_MIN_ELEMENT_HPP
