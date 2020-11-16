#ifndef MIGRAPHX_GUARD_OPERATORS_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SLICE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>
#include <vector>

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

    value attributes() const
    {
        value normalize     = value::object{};
        normalize["axes"]   = value::array{normalize_attribute::include_min};
        normalize["starts"] = value::array{normalize_attribute::clip_max,
                                           normalize_attribute::clip_min,
                                           normalize_attribute::include_max,
                                           normalize_attribute::use_len,
                                           normalize_attribute::include_min};
        normalize["ends"]   = value::array{normalize_attribute::clip_max,
                                         normalize_attribute::clip_min,
                                         normalize_attribute::include_max,
                                         normalize_attribute::use_len,
                                         normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "slice"; }

    void tune_attributes(std::vector<int64_t>& tuned_axes,
                         std::vector<int64_t>& tuned_starts,
                         std::vector<int64_t>& tuned_ends,
                         const std::vector<std::size_t>& lens) const
    {
        // tune axes
        int64_t n_rank = static_cast<int64_t>(lens.size());
        if(!std::all_of(tuned_axes.begin(), tuned_axes.end(), [=](auto i) {
               return (i < n_rank and i >= -n_rank);
           }))
        {
            MIGRAPHX_THROW("SLICE: input axis " + to_string_range(tuned_axes) + " out of range");
        }
        std::transform(tuned_axes.begin(), tuned_axes.end(), tuned_axes.begin(), [=](auto i) {
            return (i < 0) ? (i + n_rank) : i;
        });

        std::vector<int64_t> axis_lens(tuned_axes.size());
        std::transform(tuned_axes.begin(), tuned_axes.end(), axis_lens.begin(), [&](auto axis) {
            return lens[axis];
        });

        // tune starts
        std::transform(tuned_starts.begin(),
                       tuned_starts.end(),
                       axis_lens.begin(),
                       tuned_starts.begin(),
                       [=](auto i, auto dim) {
                           i = (i < -dim) ? -dim : ((i > dim) ? dim : i);
                           return (i < 0) ? (i + dim) : i;
                       });

        // tune ends
        std::transform(tuned_ends.begin(),
                       tuned_ends.end(),
                       axis_lens.begin(),
                       tuned_ends.begin(),
                       [=](auto i, auto dim) {
                           i = (i < -dim) ? -dim : ((i > dim) ? dim : i);
                           return (i < 0) ? (i + dim) : i;
                       });

        if(!(tuned_ends >= tuned_starts))
        {
            MIGRAPHX_THROW("SLICE: starts and ends does not match");
        }
    }

    auto fix_index(const std::vector<std::size_t>& lens, std::size_t axis, int64_t index) const
    {
        int64_t r = std::min(index, static_cast<int64_t>(lens[axis]));
        if(r < 0)
            r += lens[axis];
        return std::size_t(r);
    }

    auto compute_offset(const shape& s) const
    {
        std::vector<int64_t> tuned_axes      = axes;
        std::vector<int64_t> tuned_starts    = starts;
        std::vector<int64_t> tuned_ends      = ends;
        const std::vector<std::size_t>& lens = s.lens();
        tune_attributes(tuned_axes, tuned_starts, tuned_ends, lens);

        const std::vector<std::size_t>& strides = s.strides();
        auto offset                             = 0;
        if(!tuned_axes.empty())
        {
            for(std::size_t i = 0; i < tuned_axes.size(); i++)
            {
                auto axis = tuned_axes[i];
                offset += fix_index(lens, axis, tuned_starts[i]) * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += fix_index(lens, axis, tuned_starts[axis]) * strides[axis];
            }
        }
        return offset;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape        = inputs[0];
        auto t                  = input_shape.type();
        const auto& old_lens    = input_shape.lens();
        const auto& old_strides = input_shape.strides();

        if(std::any_of(
               axes.begin(), axes.end(), [&](auto i) { return (i >= old_lens.size() and i < 0); }))
        {
            MIGRAPHX_THROW("SLICE: input axis " + to_string_range(axes) + " out of range");
        }

        if(starts.size() != axes.size() || axes.size() != ends.size())
        {
            MIGRAPHX_THROW("SLICE: inconsistent sizes");
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
