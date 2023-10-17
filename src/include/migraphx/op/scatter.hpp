/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/name.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/argument.hpp>

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
        check_shapes{inputs, *this}.has(3);
        // If non-packed, this converts to a packed output while preserving permutation of tensor
        return inputs.front().with_lens(inputs.front().lens());
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto& self = static_cast<const Derived&>(*this);

        // max dimension in each axis
        auto axis_dim_size = output_shape.lens()[axis];
        // cast all arguments as correct type
        visit_all(result, args[0], args[2])([&](auto output, auto data, auto update) {
            // copy all of data to output
            std::copy(data.begin(), data.end(), output.begin());
            args[1].visit([&](auto indices) {
                auto ind_s = indices.get_shape();
                // iterate through items in shape
                shape_for_each(ind_s, [&](const auto& idx) {
                    auto out_idx = idx;

                    // Overloaded tensor_view::() invokes indexing logic of
                    // std::size_t shape::index(std::size_t i) const
                    // which handles nonstandard shapes correctly
                    auto index = indices(idx.begin(), idx.end());

                    // normalize negative indexes (may be redundant after using
                    // normalize_compute_shape())
                    index         = (index < 0) ? index + axis_dim_size : index;
                    out_idx[axis] = index;

                    // look up the appropriate locations in output, using idx and out_idx.
                    // call reduction() method of derived struct to copy and reduce that element
                    self.reduction()(output(out_idx.begin(), out_idx.end()),
                                     update(idx.begin(), idx.end()));
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
