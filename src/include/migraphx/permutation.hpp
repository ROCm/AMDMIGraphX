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
    auto tuned_dims = dims;
    if(tuned_dims.size() == 0)
    {
        tuned_dims.resize(permutation.size());
        std::iota(tuned_dims.begin(), tuned_dims.end(), 0);
        std::reverse(tuned_dims.begin(), tuned_dims.end());
    }
    Vector result(tuned_dims.size());
    assert(tuned_dims.size() == permutation.size());
    for(std::size_t i = 0; i < tuned_dims.size(); i++)
    {
        result[i] = tuned_dims[permutation[i]];
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

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation);

std::vector<int64_t> find_permutation(const shape& s);
std::vector<int64_t> find_permutation(const std::vector<shape>& shapes);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
