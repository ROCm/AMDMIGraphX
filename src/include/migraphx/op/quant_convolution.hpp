#ifndef MIGRAPHX_GUARD_OPERATORS_QUANT_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANT_CONVOLUTION_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/value.hpp>
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

    value attributes() const
    {
        return {{"general_data_type", "convolution"}, {"normalize_padding", "padding"}};
    }

    std::string name() const { return "quant_convolution"; }

    void check_attribute_size() const
    {
        if(not((padding.size() == stride.size() or (padding.size() / 2) == stride.size()) and
               stride.size() == dilation.size()))
        {
            MIGRAPHX_THROW("QUANT_CONVOLUTION: inconsistent attribute sizes");
        }
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);
        check_attribute_size();

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        size_t kdims         = input.lens().size() - 2;
        if(kdims != this->kdims())
        {
            MIGRAPHX_THROW("quant_convolution: input k-dims does not match attribute size");
        }

        // all input type must be int8_type and output is float_type
        if(t != shape::int8_type)
        {
            MIGRAPHX_THROW("QUANT_CONVOLUTION: only accept input and weights of type int8_t");
        }
        t = shape::int32_type;

        std::vector<size_t> output_lens{input.lens()[0], weights.lens()[0]};
        auto padding_size = padding.size();
        for(size_t i = 0; i < kdims; i++)
        {
            auto padding_factor = 2 * padding[i];
            if(padding_size == 2 * kdims)
                padding_factor = padding[i] + padding[i + kdims];
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                (input.lens()[i + 2] - (1 + dilation[i] * (weights.lens()[i + 2] - 1)) +
                 padding_factor) /
                        stride[i] +
                    1)));
        }

        return inputs[0].with_lens(t, output_lens);
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
