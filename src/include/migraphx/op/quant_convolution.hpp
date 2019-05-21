#ifndef MIGRAPHX_GUARD_OPERATORS_QUANT_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANT_CONVOLUTION_HPP

#include <array>
#include <migraphx/op/common.hpp>
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

struct quant_convolution
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};

    padding_mode_t padding_mode = default_;
    int group                   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.group, "group"));
    }

    std::string name() const { return "quant_convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().only_dims(4);

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();

        // all input type must be int8_type and output is float_type
        if(t != shape::int8_type)
        {
            MIGRAPHX_THROW("QUANT_CONVOLUTION: only accept input and weights of type int8_t");
        }
        t = shape::float_type;

        if(padding_mode == default_)
        {
            return {t,
                    {
                        input.lens()[0],
                        weights.lens()[0],
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            (input.lens()[2] - (1 + dilation[0] * (weights.lens()[2] - 1)) +
                             2 * padding[0]) /
                                    stride[0] +
                                1)),
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            (input.lens()[3] - (1 + dilation[1] * (weights.lens()[3] - 1)) +
                             2 * padding[1]) /
                                    stride[1] +
                                1)),
                    }};
        }
        else if(padding_mode == same)
        {
            return {t,
                    {input.lens()[0],
                     weights.lens()[0],
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[2]) / stride[0])),
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[3]) / stride[1]))}};
        }
        else if(padding_mode == valid)
        {
            return {
                t,
                {input.lens()[0],
                 weights.lens()[0],
                 static_cast<std::size_t>(std::ceil(
                     static_cast<double>(input.lens()[2] - weights.lens()[2] + 1) / stride[0])),
                 static_cast<std::size_t>(std::ceil(
                     static_cast<double>(input.lens()[3] - weights.lens()[3] + 1) / stride[1]))}};
        }
        else
        {
            MIGRAPHX_THROW("QUANT_CONVOLUTION: invalid padding mode");
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
