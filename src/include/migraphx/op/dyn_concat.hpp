/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_DYN_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_DYN_CONCAT_HPP

#include <migraphx/argument.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/par.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Concatenate valid prefixes of maximum-capacity inputs.
 *
 * Inputs are ordered as:
 *   data_0, ..., data_n, num_elements_0, ..., num_elements_n
 *
 * Each num_elements input is an int64 tensor with shape {1} and gives the valid extent of the
 * corresponding data input along axis. The output is a tuple containing a zero-padded static
 * maximum-capacity buffer and the sum of the valid extents.
 */
struct dyn_concat
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

    std::string name() const { return "dyn_concat"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty() or inputs.size() % 2 != 0)
            MIGRAPHX_THROW("DYN_CONCAT: expected an equal number of data and count inputs");
        const auto n = inputs.size() / 2;
        check_shapes{inputs.begin(), inputs.begin() + n, *this, true}.same_ndims().same_type();

        const auto rank = inputs.front().ndim();
        if(axis < 0 or static_cast<std::size_t>(axis) >= rank)
            MIGRAPHX_THROW("DYN_CONCAT: axis is out of range");
        const auto normalized_axis = static_cast<std::size_t>(axis);

        const shape count_shape{shape::int64_type, {1}};
        if(not std::all_of(
               inputs.begin() + n, inputs.end(), [&](const shape& s) { return s == count_shape; }))
            MIGRAPHX_THROW("DYN_CONCAT: count inputs must be int64 tensors with shape {1}");

        const std::vector<shape> data_shapes(inputs.begin(), inputs.begin() + n);
        const auto dynamic_inputs = shape::to_dynamic(data_shapes);
        const auto& first_dims    = dynamic_inputs.front().dyn_dims();
        if(not migraphx::all_of(range(rank), [&](std::size_t current_axis) {
               if(current_axis == normalized_axis)
                   return true;
               const auto& first_dim = first_dims.at(current_axis);
               return first_dim.is_fixed() and
                      std::all_of(
                          dynamic_inputs.begin(), dynamic_inputs.end(), [&](const shape& s) {
                              return s.dyn_dims().at(current_axis) == first_dim;
                          });
           }))
            MIGRAPHX_THROW("DYN_CONCAT: non-concat dimensions must be fixed and equal");

        auto output_lens                = inputs.front().max_lens();
        output_lens.at(normalized_axis) = std::accumulate(
            inputs.begin(), inputs.begin() + n, std::size_t{0}, [&](auto total, const shape& s) {
                const auto length = s.max_lens().at(normalized_axis);
                if(length > std::numeric_limits<std::size_t>::max() - total)
                    MIGRAPHX_THROW("DYN_CONCAT: maximum output dimension overflow");
                return total + length;
            });
        return shape{{shape{inputs.front().type(), output_lens}, shape{shape::int64_type, {1}}}};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto n               = args.size() / 2;
        const auto normalized_axis = static_cast<std::size_t>(axis);
        const auto& output_shapes  = output_shape.sub_shapes();
        argument output{output_shapes.front()};
        argument total_output{output_shapes.back()};

        output.visit([](auto values) {
            using value_type = typename decltype(values)::value_type;
            std::fill(values.begin(), values.end(), value_type{0});
        });

        auto indices = range(n);
        std::vector<std::size_t> counts(n);
        std::transform(indices.begin(), indices.end(), counts.begin(), [&](std::size_t i) {
            const auto count_values = args.at(n + i).to_vector<int64_t>();
            if(count_values.front() < 0)
                MIGRAPHX_THROW("DYN_CONCAT: count cannot be negative");
            const auto count = static_cast<std::size_t>(count_values.front());
            if(count > args.at(i).get_shape().lens().at(normalized_axis))
                MIGRAPHX_THROW("DYN_CONCAT: count exceeds the input dimension");
            return count;
        });
        std::vector<std::size_t> offsets(n);
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), std::size_t{0});
        const auto output_offset = std::accumulate(counts.begin(), counts.end(), std::size_t{0});
        if(output_offset > output_shapes.front().lens().at(normalized_axis))
            MIGRAPHX_THROW("DYN_CONCAT: counts exceed the output capacity");

        migraphx::for_each(
            indices.begin(),
            indices.end(),
            offsets.begin(),
            [&](std::size_t i, std::size_t offset) {
                const auto& input_shape          = args.at(i).get_shape();
                auto slice_lens                  = input_shape.lens();
                slice_lens.at(normalized_axis)   = counts.at(i);
                auto output_start                = std::vector<std::size_t>(slice_lens.size(), 0);
                output_start.at(normalized_axis) = offset;
                const auto output_element_offset = output_shapes.front().index(output_start);

                visit_all(output, args.at(i))([&](auto output_view, auto input_view) {
                    auto input_slice =
                        make_view(shape{input_shape.type(), slice_lens, input_shape.strides()},
                                  input_view.data());
                    auto output_slice = make_view(shape{output_shapes.front().type(),
                                                        slice_lens,
                                                        output_shapes.front().strides()},
                                                  output_view.data() + output_element_offset);
                    std::copy(input_slice.begin(), input_slice.end(), output_slice.begin());
                });
            });
        total_output.visit([&](auto total) {
            total.front() = static_cast<typename decltype(total)::value_type>(output_offset);
        });
        return {{output, total_output}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
