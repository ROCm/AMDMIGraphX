#ifndef ROCM_GUARD_ROCM_ALGORITHM_FIND_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_FIND_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/find_if.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class T>
constexpr Iterator find(Iterator first, Iterator last, const T& value)
{
    return find_if(first, last, [&](const auto& x) { return x == value; });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_FIND_HPP
