#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP

#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline std::size_t calculate_padding(std::size_t weight_dim, std::size_t dilation)
{
    return (dilation * (weight_dim - 1)) / 2;
}

inline void calculate_padding(int64_t idx, std::vector<int64_t>& pads, int64_t input_dim, int64_t stride, int64_t dilation, int64_t weight_dim)
{
    int64_t output_dim = input_dim / stride;
    int64_t pad = std::max(static_cast<int64_t>(0), (output_dim - 1) * stride + dilation * weight_dim - input_dim);
    pads[idx] = pad / 2;
    pads[idx + 2] = pad - pad / 2;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
