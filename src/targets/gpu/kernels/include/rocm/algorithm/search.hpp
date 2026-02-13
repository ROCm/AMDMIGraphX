#ifndef ROCM_GUARD_ROCM_ALGORITHM_SEARCH_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_SEARCH_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator1, class Iterator2, class BinaryPredicate>
constexpr Iterator1
search(Iterator1 first, Iterator1 last, Iterator2 s_first, Iterator2 s_last, BinaryPredicate pred)
{
    for(;; ++first)
    {
        Iterator1 it = first;
        for(Iterator2 s_it = s_first;; ++it, ++s_it)
        {
            if(s_it == s_last)
            {
                return first;
            }
            if(it == last)
            {
                return last;
            }
            if(not pred(*it, *s_it))
            {
                break;
            }
        }
    }
}

template <class Iterator1, class Iterator2>
constexpr Iterator1 search(Iterator1 first, Iterator1 last, Iterator2 s_first, Iterator2 s_last)
{
    return search(first, last, s_first, s_last, [](auto&& x, auto&& y) { return x == y; });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_SEARCH_HPP
