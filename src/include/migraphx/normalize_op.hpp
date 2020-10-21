#ifndef MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct operation;

template<class T, class...>
struct select_dependent_type { using type = T; };
template<class T, class... Ts>
using dependent_type = typename select_dependent_type<T, Ts...>::type;



void normalize_op(operation& op, std::vector<shape> inputs);

template<class T>
shape normalize_compute_shape_op(T&& x, std::vector<shape> inputs)
{
    dependent_type<operation, T> y = x;
    normalize_op(y, inputs);
    return any_cast<T>(y).normalize_compute_shape(inputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

