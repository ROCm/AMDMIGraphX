#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_DIMS_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_DIMS_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<shape> reduce_dims(const std::vector<shape>& shapes);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
