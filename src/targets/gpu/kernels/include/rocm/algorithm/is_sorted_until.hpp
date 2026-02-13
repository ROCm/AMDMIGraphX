#ifndef ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_UNTIL_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_UNTIL_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr Iterator is_sorted_until(Iterator first, Iterator last, Compare comp)
{
    if(first != last)
    {
        Iterator next = first;
        while(++next != last)
        {
            if(comp(*next, *first))
                return next;
            first = next;
        }
    }
    return last;
}

template <class Iterator>
constexpr Iterator is_sorted_until(Iterator first, Iterator last)
{
    return is_sorted_until(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_UNTIL_HPP
