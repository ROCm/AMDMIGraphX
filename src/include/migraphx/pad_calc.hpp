#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP

#include <utility>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline void calculate_padding(int64_t idx,
                              std::vector<int64_t>& pads,
                              int64_t input_dim,
                              int64_t stride,
                              int64_t dilation,
                              int64_t weight_dim,
                              bool is_same_upper = true)
{
    int64_t output_dim     = (input_dim + stride - 1) / stride; // round up result
    int64_t new_weight_dim = weight_dim + (weight_dim - 1) * (dilation - 1);
    int64_t pad =
        std::max(static_cast<int64_t>(0), (output_dim - 1) * stride + new_weight_dim - input_dim);

    if(is_same_upper)
    {
        pads[idx]     = pad / 2;
        pads[idx + 2] = pad - pad / 2;
    }
    else
    {
        pads[idx + 2] = pad / 2;
        pads[idx]     = pad - pad / 2;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
