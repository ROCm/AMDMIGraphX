#ifndef ROCM_GUARD_ROCM_ALGORITHM_ITER_SWAP_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_ITER_SWAP_HPP

#include <rocm/config.hpp>
#include <rocm/utility/swap.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator1, class Iterator2>
constexpr void iter_swap(Iterator1 a, Iterator2 b)
{
    if(a == b)
        return;
    swap(*a, *b);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_ITER_SWAP_HPP
