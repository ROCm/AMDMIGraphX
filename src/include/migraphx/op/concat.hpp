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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONCAT_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/permutation.hpp>
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
        auto n_dims = args[0].get_shape().lens().size();
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> offset(n_dims, 0);
        offset[axis] = 0;
        for(const auto& arg : args)
        {
            offsets.push_back(output_shape.index(offset));
            offset[axis] += arg.get_shape().lens()[axis];
        }
        return offsets;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        // inputs can contain 1 or more shapes (variadic).  compute_shape_op ensures there must
        // be at least 1.
        check_shapes{inputs, *this, true}.same_ndims().same_type();

        bool all_static =
            std::none_of(inputs.begin(), inputs.end(), [](const shape& s) { return s.dynamic(); });
        auto unified = shape::to_dynamic(inputs);

        const auto& dds0 = unified.front().dyn_dims();
        for(std::size_t i = 0; i < dds0.size(); ++i)
        {
            if(i == axis)
                continue;
            if(not std::all_of(unified.begin(), unified.end(), [&](const shape& s) {
                   return s.dyn_dims()[i] == dds0[i];
               }))
                MIGRAPHX_THROW("CONCAT: all input dimensions should match in axis " +
                               std::to_string(i));
        }

        auto new_dds  = dds0;
        new_dds[axis] = std::accumulate(
            unified.begin() + 1, unified.end(), dds0[axis], [&](const auto& acc, const shape& s) {
                return acc + s.dyn_dims()[axis];
            });

        auto type = unified.front().type();
        if(all_static)
        {
            std::vector<std::size_t> new_lens(new_dds.size());
            std::transform(new_dds.begin(), new_dds.end(), new_lens.begin(), [](const auto& d) {
                assert(d.sym_expr.is_literal());
                return d.sym_expr.eval_uint({});
            });
            return shape::from_permutation(type, new_lens, find_permutation(inputs));
        }
        if(unified.front().symbolic())
            return shape::from_permutation(type, new_dds, find_permutation(unified));
        return {type, new_dds};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        std::vector<std::size_t> coffsets = compute_offsets(dyn_out.computed_shape, args);
        for(std::size_t l = 0; l < args.size(); l++)
        {
            auto argl = args[l];
            visit_all(result, argl)([&](auto output, auto input) {
                auto slice_shape = shape{dyn_out.computed_shape.type(),
                                         input.get_shape().lens(),
                                         dyn_out.computed_shape.strides()};
                auto slice       = make_view(slice_shape, output.data() + coffsets[l]);
                std::copy(input.begin(), input.end(), slice.begin());
            });
        }
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
