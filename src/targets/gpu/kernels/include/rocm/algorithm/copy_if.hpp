#ifndef ROCM_GUARD_ROCM_ALGORITHM_COPY_IF_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_COPY_IF_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt, class OutputIt, class UnaryPredicate>
constexpr OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPredicate pred)
{
    for(; first != last; ++first)
    {
        if(pred(*first))
        {
            *d_first = *first;
            ++d_first;
        }
    }
    return d_first;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_COPY_IF_HPP
