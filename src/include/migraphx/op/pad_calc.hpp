#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP

#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

size_t calculate_padding(size_t weight_dim, size_t dilation)
{
    return (dilation * (weight_dim - 1)) / 2;
}

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
