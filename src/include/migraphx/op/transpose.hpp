#ifndef MIGRAPHX_GUARD_OPERATORS_TRANSPOSE_HPP
#define MIGRAPHX_GUARD_OPERATORS_TRANSPOSE_HPP

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

struct transpose
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    std::string name() const { return "transpose"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto input         = inputs.at(0);
        auto input_lens    = input.lens();
        auto input_strides = input.strides();
        auto t             = input.type();
        auto tuned_dims    = dims;
        // if not perm provided, reverse the dims
        if(tuned_dims.empty())
        {
            tuned_dims.resize(input_lens.size());
            std::iota(tuned_dims.begin(), tuned_dims.end(), 0);
            std::reverse(tuned_dims.begin(), tuned_dims.end());
        }

        if(tuned_dims.size() != input_lens.size())
        {
            MIGRAPHX_THROW("Permutation has wrong number of axes");
        }
        std::vector<int64_t> axes(tuned_dims.size());
        std::iota(axes.begin(), axes.end(), 0);
        if(!std::is_permutation(axes.begin(), axes.end(), tuned_dims.begin()))
        {
            MIGRAPHX_THROW("Invalid permutation");
        }
        std::vector<size_t> output_lens(input_lens.size());
        std::vector<size_t> output_strides(input_lens.size());
        for(std::size_t i = 0; i < output_lens.size(); i++)
        {
            output_lens[i]    = input_lens[tuned_dims[i]];
            output_strides[i] = input_strides[tuned_dims[i]];
        }
        return {t, output_lens, output_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
