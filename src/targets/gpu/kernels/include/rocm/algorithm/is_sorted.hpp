#ifndef ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_HPP

#include <rocm/config.hpp>
#include <rocm/functional/operations.hpp>
#include <rocm/algorithm/is_sorted_until.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class Compare>
constexpr bool is_sorted(Iterator first, Iterator last, Compare comp)
{
    return is_sorted_until(first, last, comp) == last;
}

template <class Iterator>
constexpr bool is_sorted(Iterator first, Iterator last)
{
    return is_sorted(first, last, less<>{});
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_IS_SORTED_HPP
