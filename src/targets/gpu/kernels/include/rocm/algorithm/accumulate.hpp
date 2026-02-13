#ifndef ROCM_GUARD_ROCM_ALGORITHM_ACCUMULATE_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_ACCUMULATE_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt, class T, class BinaryOperation>
constexpr T accumulate(InputIt first, InputIt last, T init, BinaryOperation op)
{
    for(; first != last; ++first)
    {
        init = op(static_cast<T&&>(init), *first);
    }
    return init;
}

template <class InputIt, class T>
constexpr T accumulate(InputIt first, InputIt last, T init)
{
    return accumulate(first, last, init, [](auto x, auto y) { return x + y; });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_ACCUMULATE_HPP
