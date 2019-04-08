#ifndef MIGRAPHX_GUARD_OPERATORS_IM2COL_HPP
#define MIGRAPHX_GUARD_OPERATORS_IM2COL_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct im2col
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};

    padding_mode_t padding_mode = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"));
    }

    std::string name() const { return "im2col"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input          = inputs[0];
        auto weights        = inputs[1];
        auto batch_size     = input.lens()[0];
        auto input_channels = weights.lens()[1];
        auto kernel_height  = weights.lens()[2];
        auto kernel_width   = weights.lens()[3];
        check_shapes{inputs, *this}.has(2);
        if(batch_size != 1)
            MIGRAPHX_THROW("im2col only support batch_size 1");
        auto output_height = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[2] - (1 + dilation[0] * (kernel_height - 1)) + 2 * padding[0]) /
                    stride[0] +
                1));
        auto output_width  = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[3] - (1 + dilation[1] * (kernel_width - 1)) + 2 * padding[1]) /
                    stride[1] +
                1));
        auto channels_col  = kernel_height * kernel_width * input_channels;
        return {input.type(), {output_height * output_width, channels_col}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
