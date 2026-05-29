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
#include <limits>
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

        if(std::none_of(inputs.begin(), inputs.end(), [&](const shape& s) { return s.dynamic(); }))
        {
            // Static input shapes
            const auto& first_shape_lens = inputs.front().lens();
            const auto& type             = inputs.front().type();
            for(std::size_t ll = 0; ll < first_shape_lens.size(); ll++)
            {
                if(ll != axis)
                {
                    if(not std::all_of(inputs.begin(), inputs.end(), [&](auto s) {
                           return s.lens()[ll] == first_shape_lens[ll];
                       }))
                    {
                        MIGRAPHX_THROW("CONCAT: all input dimensions should match along axis " +
                                       std::to_string(ll));
                    }
                }
            }
            std::size_t new_dim_axis = 0;
            for(const auto& input : inputs)
            {
                const auto& lens = input.lens();
                new_dim_axis += lens[axis];
            }
            std::vector<std::size_t> new_lens = first_shape_lens;
            new_lens[axis]                    = new_dim_axis;
            return shape::from_permutation(type, new_lens, find_permutation(inputs));
        }
        else if(std::all_of(
                    inputs.begin(), inputs.end(), [&](const shape& s) { return s.dynamic(); }))
        {
            // Dynamic input shapes. On non-concat axes the dims must agree,
            // but a fully-unconstrained dim {0, SIZE_MAX} is a wildcard: it
            // is the output of an op whose shape is only known at runtime
            // (e.g. broadcast_with_dims / ONNX Expand whose target shape is
            // computed from another tensor's shape), so it matches any dim
            // and adopts that dim's constraint in the output.
            auto is_unconstrained = [](const shape::dynamic_dimension& dd) {
                const auto iv = dd.get_interval();
                return iv.min == 0 and iv.max == std::numeric_limits<std::size_t>::max();
            };
            auto out_dims = inputs[0].dyn_dims();
            for(std::size_t index = 0; index < inputs[0].ndim(); index++)
            {
                if(index == axis)
                    continue;
                for(const auto& s : inputs)
                {
                    const auto& dd = s.dyn_dims()[index];
                    if(is_unconstrained(out_dims[index]))
                        out_dims[index] = dd;
                    else if(not is_unconstrained(dd) and dd != out_dims[index])
                        MIGRAPHX_THROW("CONCAT: all input dimensions should match in axis " +
                                       std::to_string(index));
                }
            }
            // Sum the concat axis. If any input is unconstrained on the
            // axis, the sum is unknown too -- emit a wildcard rather than
            // overflowing SIZE_MAX. The exact size is recovered once the
            // program is specialised and the runtime op is constant-folded.
            constexpr std::size_t size_max = std::numeric_limits<std::size_t>::max();
            std::size_t new_min            = 0;
            std::size_t new_max            = 0;
            bool axis_unconstrained        = false;
            for(const auto& input : inputs)
            {
                const auto& ddim = input.dyn_dims()[axis];
                if(is_unconstrained(ddim))
                {
                    axis_unconstrained = true;
                    break;
                }
                const auto dim_interval = ddim.get_interval();
                new_min += dim_interval.min;
                new_max += dim_interval.max;
            }

            out_dims[axis] = axis_unconstrained
                                 ? migraphx::shape::dynamic_dimension{0, size_max}
                                 : migraphx::shape::dynamic_dimension{new_min, new_max};
            return {inputs[0].type(), out_dims};
        }
        else
        {
            MIGRAPHX_THROW("CONCAT: Cannot mix static and dynamic input shapes.");
        }
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
