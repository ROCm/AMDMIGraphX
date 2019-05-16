#ifndef MIGRAPHX_GUARD_OPERATORS_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SLICE_HPP

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

struct slice
{
    std::vector<int64_t> axes;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.starts, "starts"), f(self.ends, "ends"));
    }

    std::string name() const { return "slice"; }

    auto fix_index(const std::vector<std::size_t>& lens, std::size_t axis, int64_t index) const
    {
        int64_t r = std::min(index, static_cast<int64_t>(lens[axis]));
        if(r < 0)
            r += lens[axis];
        return std::size_t(r);
    }

    auto compute_offset(const shape& s) const
    {
        const std::vector<std::size_t>& lens    = s.lens();
        const std::vector<std::size_t>& strides = s.strides();
        auto offset                             = 0;
        if(!axes.empty())
        {
            for(std::size_t i = 0; i < axes.size(); i++)
            {
                auto axis = axes[i];
                offset += fix_index(lens, axis, starts[i]) * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += fix_index(lens, axis, starts[axis]) * strides[axis];
            }
        }
        return offset;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape        = inputs[0];
        auto t                  = input_shape.type();
        const auto& old_lens    = input_shape.lens();
        const auto& old_strides = input_shape.strides();
        if(starts.size() != axes.size() || axes.size() != ends.size())
        {
            MIGRAPHX_THROW("inconsistent sizes");
        }
        std::vector<std::size_t> new_lens = old_lens;
        for(std::size_t i = 0; i < axes.size(); i++)
        {
            auto axis = axes[i];
            new_lens[axis] =
                fix_index(old_lens, axis, ends[i]) - fix_index(old_lens, axis, starts[i]);
        }
        return shape{t, new_lens, old_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        auto input  = args[0];
        auto offset = compute_offset(input.get_shape()) * output_shape.type_size();
        return {std::move(output_shape), [=] { return input.data() + offset; }};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
