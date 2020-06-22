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
    std::vector<std::size_t> padding  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

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

    void check_attribute_size() const
    {
        if(not(padding.size() == stride.size() and padding.size() == dilation.size()))
        {
            MIGRAPHX_THROW("quant_convolution: inconsistent attribute sizes");
        }
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);
        check_attribute_size();

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        size_t kdims         = input.lens().size() - 2;

        // all input type must be int8_type and output is float_type
        if(t != shape::int8_type)
        {
            MIGRAPHX_THROW("QUANT_CONVOLUTION: only accept input and weights of type int8_t");
        }
        t = shape::int32_type;

        std::vector<size_t> output_lens{input.lens()[0], weights.lens()[0]};

        for(size_t i = 0; i < kdims; i++)
        {
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                (input.lens()[i + 2] - (1 + dilation[i] * (weights.lens()[i + 2] - 1)) +
                 2 * padding[i]) /
                        stride[i] +
                    1)));
        }

        return {t, output_lens};
    }

    size_t kdims() const
    {
        check_attribute_size();
        return padding.size();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
