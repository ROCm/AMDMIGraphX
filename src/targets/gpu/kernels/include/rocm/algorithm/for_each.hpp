#ifndef ROCM_GUARD_ROCM_ALGORITHM_FOR_EACH_HPP
#define ROCM_GUARD_ROCM_ALGORITHM_FOR_EACH_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class Iterator, class F>
constexpr F for_each(Iterator first, Iterator last, F f)
{
    for(; first != last; ++first)
    {
        f(*first);
    }
    return f;
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ALGORITHM_FOR_EACH_HPP
