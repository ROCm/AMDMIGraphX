#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatter
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

    std::string name() const { return "scatter"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        // max dimension in axis
        auto axis_dim_size = output_shape.lens()[axis];
        visit_all(result, args[0], args[2])([&](auto output, auto data, auto update) {
            std::copy(data.begin(), data.end(), output.begin());
            args[1].visit([&](auto indices) {
                auto ind_s = indices.get_shape();
                shape_for_each(ind_s, [&](const auto& idx) {
                    auto out_idx  = idx;
                    auto index    = indices[ind_s.index(idx)];
                    index         = (index < 0) ? index + axis_dim_size : index;
                    out_idx[axis] = index;
                    output[output_shape.index(out_idx)] = update[ind_s.index(idx)];
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
