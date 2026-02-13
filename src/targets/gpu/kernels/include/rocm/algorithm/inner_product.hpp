#ifndef ROCM_GUARD_ROCM_ALGORITHM_INNER_PRODUCT_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_INNER_PRODUCT_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class InputIt1, class InputIt2, class T, class BinaryOperation1, class BinaryOperation2>
constexpr T inner_product(InputIt1 first1,
                          InputIt1 last1,
                          InputIt2 first2,
                          T init,
                          BinaryOperation1 op1,
                          BinaryOperation2 op2)
{
    while(first1 != last1)
    {
        init = op1(init, op2(*first1, *first2));
        ++first1;
        ++first2;
    }
    return init;
}

template <class InputIt1, class InputIt2, class T>
constexpr T inner_product(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init)
{
    return inner_product(
        first1,
        last1,
        first2,
        init,
        [](auto x, auto y) { return x + y; },
        [](auto x, auto y) { return x * y; });
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_INNER_PRODUCT_HPP
