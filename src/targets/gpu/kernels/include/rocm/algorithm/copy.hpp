#ifndef ROCM_GUARD_ROCM_ALGORITHM_COPY_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_COPY_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first)
{
    while(first != last)
    {
        *d_first++ = *first++;
    }
    return d_first;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_COPY_HPP
