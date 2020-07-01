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
    std::string mode                 = "average";
    std::vector<std::size_t> padding = {0, 0};
    std::vector<std::size_t> stride  = {1, 1};
    std::vector<std::size_t> lengths = {1, 1};
    padding_mode_t padding_mode      = default_;

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

    void check_attribute_size() const
    {
        if(not(padding.size() == stride.size() and padding.size() == lengths.size()))
        {
            MIGRAPHX_THROW("pooling: inconsistent attribute sizes");
        }
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);

        const shape& input = inputs.at(0);
        auto t             = input.type();

        auto input_lens = input.lens();
        size_t kdims    = input_lens.size() - 2;
        if(kdims != this->kdims())
        {
            MIGRAPHX_THROW("pooling: input k-dims does not match attribute size");
        }

        std::vector<std::size_t> output_lens(input_lens.begin(), input_lens.begin() + 2);

        for(size_t i = 0; i < kdims; i++)
        {
            assert(lengths[i] <= input_lens[i + 2] + 2 * padding[i]);

            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                floor_divide<std::ptrdiff_t>(input_lens[i + 2] + 2 * padding[i] - lengths[i],
                                             stride[i]) +
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
