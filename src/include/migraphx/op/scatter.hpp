#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/name.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// The scatter operator fetches a subset of data given by an index array and then performs a
// reduction operation (add, multiply, or just set the data) on each element returned.  We implement
// it as a separate derived struct for each of the three reduction methods.  The related operator
// scatterND is a generalization that works on a set of 3 tensors of different ranks.  The
// complementary operations are gather/gatherND.
//
// This is a template for deriving child structs from.  Each child needs to define
// only a reduction() method.  Names are automatically handled by the op_name template.

template <class Derived>
struct scatter : op_name<Derived>
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

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        // If non-packed, this converts to a packed output while preserving permutation of tensor
        return inputs.front().with_lens(inputs.front().lens());
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto& self = static_cast<const Derived&>(*this);

        // max dimension in each axis
        auto axis_dim_size = output_shape.lens()[axis];
        // iterate through all element locations in data/update/output
        visit_all(result, args[0], args[2])([&](auto output, auto data, auto update) {
            // copy all of data to output
            std::copy(data.begin(), data.end(), output.begin());
            args[1].visit([&](auto indices) {
                auto ind_s = indices.get_shape();
                // iterate through items in index
                shape_for_each(ind_s, [&](const auto& idx) {
                    auto out_idx = idx;
                    auto index   = indices[ind_s.index(idx)];
                    // normalize negative indexes (may be redundant after using normalize_compute_shape())
                    index         = (index < 0) ? index + axis_dim_size : index;
                    out_idx[axis] = index;

                    // look up the appropriate location in output, using index.
                    // call reduction() method of derived struct to copy and reduce that element
                    self.reduction()(output[output_shape.index(out_idx)], update[ind_s.index(idx)]);
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
