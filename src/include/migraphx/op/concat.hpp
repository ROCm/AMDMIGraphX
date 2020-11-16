#ifndef MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct concat
{
    int64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "concat"; }
    std::vector<std::size_t> compute_offsets(const shape& output_shape,
                                             const std::vector<argument>& args) const
    {
        auto n_dims            = args[0].get_shape().lens().size();
        std::size_t axis_index = (axis < 0) ? axis + n_dims : axis;
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> offset(n_dims, 0);
        offset[axis_index] = 0;
        for(const auto& arg : args)
        {
            offsets.push_back(output_shape.index(offset));
            offset[axis_index] += arg.get_shape().lens()[axis_index];
        }
        return offsets;
    }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
        {
            MIGRAPHX_THROW("CONCAT: Number of input tensors should exceed 0");
        }

        const auto& first_shape_lens = inputs.front().lens();
        const auto& type             = inputs.front().type();
        for(std::size_t l = 0; l < first_shape_lens.size(); l++)
        {
            if(l != axis)
            {
                if(!std::all_of(inputs.begin(), inputs.end(), [&](auto s) {
                       return s.lens()[l] == first_shape_lens[l];
                   }))
                {
                    MIGRAPHX_THROW("CONCAT: Non-axis dimensions should match");
                }
            }
        }
        std::size_t new_dim_axis = 0;
        for(const auto& input : inputs)
        {
            const auto& lens = input.lens();
            new_dim_axis += lens[axis];
        }
        std::vector<std::size_t> new_lens;
        std::copy(first_shape_lens.begin(), first_shape_lens.end(), std::back_inserter(new_lens));
        new_lens[axis] = new_dim_axis;
        return {type, new_lens};
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        std::vector<std::size_t> coffsets = compute_offsets(output_shape, args);
        for(std::size_t l = 0; l < args.size(); l++)
        {
            auto argl             = args[l];
            std::size_t nelements = argl.get_shape().elements();
            visit_all(result, argl)([&](auto output, auto input) {
                auto slice_shape =
                    shape{output_shape.type(), input.get_shape().lens(), output_shape.strides()};
                auto slice = make_view(slice_shape, output.data() + coffsets[l]);
                // cppcheck-suppress useStlAlgorithm
                for(std::size_t i = 0; i < nelements; i++)
                {
                    slice[i] = input[i];
                }
            });
        }
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
