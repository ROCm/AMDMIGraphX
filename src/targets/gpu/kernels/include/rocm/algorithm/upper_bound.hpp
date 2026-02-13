#ifndef ROCM_GUARD_ROCM_ALGORITHM_UPPER_BOUND_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_UPPER_BOUND_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class T, class Compare>
constexpr Iterator upper_bound(Iterator first, Iterator last, const T& value, Compare comp)
{
    auto count = last - first;

    while(count > 0)
    {
        // NOLINTNEXTLINE(readability-qualified-auto)
        auto it   = first;
        auto step = count / 2;
        it += step;

        if(not comp(value, *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }

    return first;
}

template <class Iterator, class T>
constexpr Iterator upper_bound(Iterator first, Iterator last, const T& value)
{
    return upper_bound(first, last, value, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_UPPER_BOUND_HPP
