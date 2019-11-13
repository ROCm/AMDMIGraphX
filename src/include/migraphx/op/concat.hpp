#ifndef MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP

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

struct concat
{
    int axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "concat"; }
    std::vector<std::size_t> compute_offsets(const shape& output_shape,
                                             const std::vector<argument>& args) const
    {
        std::size_t n_dims = args[0].get_shape().lens().size();
        int axis_index = (axis < 0) ? axis + n_dims : axis;
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> offset(args[0].get_shape().lens().size(), 0);
        offset[axis_index] = 0;
        for(const auto& arg : args)
        {
            offsets.push_back(output_shape.index(offset));
            offset[axis_index] += arg.get_shape().lens()[axis_index];
        }
        return offsets;
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
        {
            MIGRAPHX_THROW("Number of input tensors should exceed 0");
        }

        const auto& first_shape_lens = inputs.front().lens();
        const auto& type             = inputs.front().type();
        int axis_index = (axis < 0) ? (first_shape_lens.size() + axis) : axis;
        for(std::size_t l = 0; l < first_shape_lens.size(); l++)
        {
            if(l != axis_index)
            {
                if(!std::all_of(inputs.begin(), inputs.end(), [&](auto s) {
                       return s.lens()[l] == first_shape_lens[l];
                   }))
                {
                    MIGRAPHX_THROW("Non-axis dimensions should match");
                }
            }
        }
        std::size_t new_dim_axis = 0;
        for(const auto& input : inputs)
        {
            const auto& lens = input.lens();
            new_dim_axis += lens[axis_index];
        }
        std::vector<std::size_t> new_lens;
        std::copy(first_shape_lens.begin(), first_shape_lens.end(), std::back_inserter(new_lens));
        new_lens[axis_index] = new_dim_axis;
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
