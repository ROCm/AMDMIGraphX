#ifndef ROCM_GUARD_ROCM_ALGORITHM_ALL_OF_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_ALL_OF_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/none_of.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt, class UnaryPredicate>
constexpr bool all_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return none_of(first, last, [=](auto&& x) { return not p(x); });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_ALL_OF_HPP
