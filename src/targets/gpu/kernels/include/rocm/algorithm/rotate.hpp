#ifndef ROCM_GUARD_ROCM_ALGORITHM_ROTATE_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_ROTATE_HPP

#include <rocm/config.hpp>
#include <rocm/algorithm/iter_swap.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator>
constexpr Iterator rotate(Iterator first, Iterator middle, Iterator last)
{
    if(first == middle)
        return last;

    if(middle == last)
        return first;

    Iterator write     = first;
    Iterator next_read = first;

    for(Iterator read = middle; read != last; ++write, ++read)
    {
        if(write == next_read)
            next_read = read;
        iter_swap(write, read);
    }

    rotate(write, next_read, last);
    return write;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_ROTATE_HPP
