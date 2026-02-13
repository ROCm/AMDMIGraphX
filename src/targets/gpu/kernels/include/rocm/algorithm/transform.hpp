#ifndef ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class OutputIterator, class UnaryOp>
constexpr OutputIterator
transform(Iterator first1, Iterator last1, OutputIterator out, UnaryOp unary_op)
{
    for(; first1 != last1; ++out, ++first1)
        *out = unary_op(*first1);

    return out;
}

template <class Iterator1, class Iterator2, class OutputIterator, class BinaryOp>
constexpr OutputIterator transform(
    Iterator1 first1, Iterator1 last1, Iterator2 first2, OutputIterator out, BinaryOp binary_op)
{
    for(; first1 != last1; ++out, ++first1, ++first2)
        *out = binary_op(*first1, *first2);

    return out;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_TRANSFORM_HPP
