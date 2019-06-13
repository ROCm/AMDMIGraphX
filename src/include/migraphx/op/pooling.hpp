#ifndef MIGRAPHX_GUARD_OPERATORS_POOLING_HPP
#define MIGRAPHX_GUARD_OPERATORS_POOLING_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/int_divide.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pooling
{
    std::string mode                   = "average";
    std::array<std::size_t, 2> padding = {{0, 0}};
    std::array<std::size_t, 2> stride  = {{1, 1}};
    std::array<std::size_t, 2> lengths = {{1, 1}};
    padding_mode_t padding_mode        = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"));
    }

    std::string name() const { return "pooling"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(4);

        const shape& input = inputs.at(0);
        auto t             = input.type();

        assert(lengths[0] <= (input.lens()[2] + 2 * padding[0]));
        assert(lengths[1] <= (input.lens()[3] + 2 * padding[1]));

        if(padding_mode == default_)
        {
            return {t,
                    {
                        input.lens()[0],
                        input.lens()[1],
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            floor_divide<std::ptrdiff_t>(
                                input.lens()[2] + 2 * padding[0] - lengths[0], stride[0]) +
                                1)),
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            floor_divide<std::ptrdiff_t>(
                                input.lens()[3] + 2 * padding[1] - lengths[1], stride[1]) +
                                1)),
                    }};
        }
        else if(padding_mode == same)
        {
            return {t,
                    {input.lens()[0],
                     input.lens()[1],
                     ceil_divide<std::size_t>(input.lens()[2], stride[0]),
                     ceil_divide<std::size_t>(input.lens()[3], stride[1])}};
        }
        else if(padding_mode == valid)
        {
            return {
                t,
                {
                    input.lens()[0],
                    input.lens()[1],
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        floor_divide<std::ptrdiff_t>(input.lens()[2] - lengths[0], stride[0]) + 1)),
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        floor_divide<std::ptrdiff_t>(input.lens()[3] - lengths[1], stride[1]) + 1)),
                }};
        }
        else
        {
            MIGRAPHX_THROW("Invalid padding mode");
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
