#ifndef MIGRAPHX_GUARD_RTGLIB_PERMUTATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_PERMUTATION_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Vector>
inline Vector reorder_dims(const Vector& dims, const std::vector<int64_t>& permutation)
{
    Vector result(dims.size());
    assert(dims.size() == permutation.size());
    for(std::size_t i = 0; i < dims.size(); i++)
    {
        result[i] = dims[permutation[i]];
    }
    return result;
}

shape reorder_shape(const shape& s, const std::vector<int64_t>& permutation);

template <class Vector, class Op>
inline std::vector<int64_t> sort_permutation(const Vector& data, Op op)
{
    std::vector<std::int64_t> result(data.size());
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(
        result.begin(), result.end(), [&](auto x, auto y) { return op(data[x], data[y]); });
    return result;
}

/*!
 * Returns the permutation needed to apply to the shape to undo the current permutation
 */
std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation);

/*!
 * Finds the permutation most likely from a transpose operator that has been applied to the shape.
 */
std::vector<int64_t> find_permutation(const shape& s);
std::vector<int64_t> find_permutation(const std::vector<shape>& shapes);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
