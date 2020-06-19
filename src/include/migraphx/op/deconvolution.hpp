#ifndef MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP

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

struct deconvolution
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

    std::string name() const { return "deconvolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);
        if(not(padding.size() == stride.size() and padding.size() == dilation.size()))
        {
            MIGRAPHX_THROW("deconvolution: inconsistent attribute sizes");
        }

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        size_t kdims         = input.lens().size() - 2;

        std::vector<size_t> output_lens{input.lens()[0], weights.lens()[1]};

        for(size_t i = 0; i < kdims; i++)
        {
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                stride[i] * (input.lens()[i + 2] - 1) +
                    ((weights.lens()[i + 2] - 1) * dilation[i] + 1) - 2 * padding[i])));
        }
        return {t, output_lens};
        // return {t,
        //         {
        //             input.lens()[0],
        //             weights.lens()[1],
        //             std::size_t(std::max<std::ptrdiff_t>(
        //                 1,
        //                 stride[0] * (input.lens()[2] - 1) +
        //                     ((weights.lens()[2] - 1) * dilation[0] + 1) - 2 * padding[0])),
        //             std::size_t(std::max<std::ptrdiff_t>(
        //                 1,
        //                 stride[1] * (input.lens()[3] - 1) +
        //                     ((weights.lens()[3] - 1) * dilation[1] + 1) - 2 * padding[1])),
        //         }};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
