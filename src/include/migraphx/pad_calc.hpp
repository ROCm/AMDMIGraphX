#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP

#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline std::size_t calculate_padding(std::size_t weight_dim, std::size_t dilation)
{
    return (dilation * (weight_dim - 1)) / 2;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
