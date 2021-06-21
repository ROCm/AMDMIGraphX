#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int64_t k;
    int64_t axis = 0;
    bool larget = true;
    bool sorted = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), 
                    f(self.largest, "largest"), 
                    f(self.sorted, "sort"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "topk"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs.at(0).lens();
        auto type = inputs.at(0).type();

        shape s_val{type, lens};
        shape s_ind{shape::int64_type, lens};

        return {s_val, s_ind};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto vec_ss = output_shape.get
        argument res_val{output_shape};
        // negative axis means counting dimensions from back
        auto lens                 = args[0].get_shape().lens();
        std::size_t axis_dim_size = lens[axis];
        // max dimension in axis
        visit_all(result, args[0])([&](auto output, auto data) {
            args[1].visit([&](auto indices) {
                if(output_shape.scalar())
                {
                    auto in_index = indices.front();
                    in_index      = (in_index < 0) ? in_index + axis_dim_size : in_index;
                    output[0]     = data[in_index];
                }
                else
                {
                    auto out_lens  = data.get_shape().lens();
                    out_lens[axis] = indices.get_shape().elements();
                    migraphx::shape out_comp_shape{data.get_shape().type(), out_lens};
                    shape_for_each(out_comp_shape, [&](const auto& out_idx) {
                        auto data_idx  = out_idx;
                        auto in_index  = indices[data_idx[axis]];
                        in_index       = (in_index < 0) ? in_index + axis_dim_size : in_index;
                        data_idx[axis] = in_index;
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
