#ifndef MIGRAPHX_GUARD_PMR_VECTOR_HPP
#define MIGRAPHX_GUARD_PMR_VECTOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/pmr.hpp>

#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace pmr {
#if MIGRAPHX_HAS_PMR
template <class T>
using vector = std::pmr::vector<T>;
#else
template <class T>
struct vector : std::vector<T>
{
    using std::vector<T>::vector;
};
#endif

} // namespace pmr
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_PMR_VECTOR_HPP
