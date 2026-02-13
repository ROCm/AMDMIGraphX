#ifndef ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/copy.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator1, class Iterator2, class OutputIterator, class Compare>
constexpr OutputIterator merge(Iterator1 first1,
                               Iterator1 last1,
                               Iterator2 first2,
                               Iterator2 last2,
                               OutputIterator d_first,
                               Compare comp)
{
    for(; first1 != last1; ++d_first)
    {
        if(first2 == last2)
            return copy(first1, last1, d_first);

        if(comp(*first2, *first1))
        {
            *d_first = *first2;
            ++first2;
        }
        else
        {
            *d_first = *first1;
            ++first1;
        }
    }
    return copy(first2, last2, d_first);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_MERGE_HPP
