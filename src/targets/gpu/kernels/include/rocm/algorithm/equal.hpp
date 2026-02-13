#ifndef ROCM_GUARD_ROCM_ALGORITHM_EQUAL_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_EQUAL_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator1, class Iterator2, class BinaryPred>
constexpr bool equal(Iterator1 first1, Iterator1 last1, Iterator2 first2, BinaryPred p)
{
    for(; first1 != last1; ++first1, ++first2)
        if(not p(*first1, *first2))
        {
            return false;
        }
    return true;
}

template <class Iterator1, class Iterator2>
constexpr bool equal(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    return equal(first1, last1, first2, [](auto&& x, auto&& y) { return x == y; });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_EQUAL_HPP
