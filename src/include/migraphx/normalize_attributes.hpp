#ifndef MIGRAPHX_GUARD_RTGLIB_NORMALIZE_ATTRIBUTES_HPP
#define MIGRAPHX_GUARD_RTGLIB_NORMALIZE_ATTRIBUTES_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct operation;

template <class T, class...>
struct select_dependent_type
{
    using type = T;
};
template <class T, class... Ts>
using dependent_type = typename select_dependent_type<T, Ts...>::type;

bool normalize_attributes(operation& op, const std::vector<std::size_t>& lens);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
