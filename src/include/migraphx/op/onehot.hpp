/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_ONEHOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_ONEHOT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Produces a one-hot tensor.
 * Called with `axis` attribute that defaults to the last axis
 * `onehot(indices, depth, values)`;
 * `onehot(indices, values)`;
 * `indicies` as a N rank tensor of indices where value is `on_value`
 * `depth` scalar with the number of classes for the one-hot dimension
 * `values` `[off_value, on_value]`
 * `axis` which axis to add the one-hot dimension to
 * For axis=0 and rank(indices) = 2:
 * output is A[indicies[j, k], j, k] = on_value; A[i, j, k] = off_value otherwise
 */
struct onehot
{
    // note this will be automatically normalized when calling normalize_compute_shape
    int64_t axis = -1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    value attributes() const
    {
        value normalize_axes   = value::object{};
        normalize_axes["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize_axes}};
    }

    std::string name() const { return "onehot"; }

    shape normalize_compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this, true}.has(3);
        // `depth` and `values` should have static shape
        (void)check_shapes{inputs.begin() + 1, inputs.end(), *this, false};
        const auto& indices_shape = inputs[0];
        shape values_shape        = inputs[2];
        auto output_dds           = indices_shape.to_dynamic().dyn_dims();
        std::size_t max_val       = std::numeric_limits<std::size_t>::max();
        output_dds.insert(output_dds.begin() + this->axis, shape::dynamic_dimension{0, max_val});
        return {values_shape.type(), output_dds};
    }

    argument compute(const shape&, std::vector<argument> args) const
    {
        auto indices_shape = args[0].get_shape();
        int64_t depth;
        args[1].visit([&](auto d) { depth = d(0); });
        auto output_lens = indices_shape.lens();
        output_lens.insert(output_lens.begin() + this->axis, depth);
        shape output_shape{args[2].get_shape().type(), output_lens};

        argument result{output_shape};
        visit_all(result, args[2])([&](auto output, auto values) {
            auto off_value = values(0);
            auto on_value  = values(1);
            par_for(output_shape.elements(), [&](auto i) { output(i) = off_value; });
            args[0].visit([&](auto indices) {
                auto ind_s = indices.get_shape();
                shape_for_each(ind_s, [&](const auto& idx) {
                    auto out_idx = idx;
                    auto index   = indices(idx.begin(), idx.end());
                    // normalize negative indexes, will fail if index is not within [-depth,
                    // depth-1]
                    index = (index < 0) ? index + depth : index;
                    out_idx.insert(out_idx.begin() + this->axis, index);
                    output(out_idx.begin(), out_idx.end()) = on_value;
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
