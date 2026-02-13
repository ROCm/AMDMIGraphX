#ifndef ROCM_GUARD_ROCM_ALGORITHM_ANY_OF_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_ANY_OF_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/find_if.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt, class UnaryPredicate>
constexpr bool any_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return find_if(first, last, p) != last;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_ANY_OF_HPP
