#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

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

struct gather
{
    int axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "gather"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard();
        auto lens = inputs[0].lens();
        int n_dim = static_cast<int>(lens.size());
        if(axis >= n_dim || axis < -n_dim)
        {
            MIGRAPHX_THROW("Gather: axis is out of range.");
        }

        // negative axis means counting dimensions from back
        int axis_index = (axis < 0) ? (n_dim + axis) : axis;

        auto type = inputs[0].type();
        lens.erase(lens.begin() + axis_index);
        if(!inputs[1].scalar())
        {
            auto ind_lens = inputs[1].lens();
            lens.insert(lens.begin() + axis_index, ind_lens.begin(), ind_lens.end());
        }

        // for scalar output
        if(lens.empty())
        {
            return {type};
        }

        return {type, lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        // negative axis means counting dimensions from back
        auto lens      = args[0].get_shape().lens();
        int axis_index = (axis < 0) ? static_cast<int>(lens.size() + axis) : axis;

        std::size_t axis_dim_size = lens[axis_index];
        // max dimension in axis
        visit_all(result, args[0])([&](auto output, auto data) {
            args[1].visit([&](auto indices) {
                if(output_shape.scalar())
                {
                    auto in_index = indices.front();
                    in_index      = (in_index < 0) ? in_index + axis_dim_size : in_index;
                    output[0]     = data[indices.front()];
                }
                else
                {
                    auto out_lens        = data.get_shape().lens();
                    out_lens[axis_index] = indices.get_shape().elements();
                    migraphx::shape out_comp_shape{data.get_shape().type(), out_lens};
                    shape_for_each(out_comp_shape, [&](const auto& out_idx) {
                        auto data_idx        = out_idx;
                        auto in_index        = indices[data_idx[axis_index]];
                        in_index             = (in_index < 0) ? in_index + axis_dim_size : in_index;
                        data_idx[axis_index] = in_index;
                        output[out_comp_shape.index(out_idx.begin(), out_idx.end())] =
                            data(data_idx.begin(), data_idx.end());
                    });
                }
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
