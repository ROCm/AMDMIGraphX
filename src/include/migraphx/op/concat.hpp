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
#include <migraphx/common.hpp>
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

    void handle_mixed_shapes(std::vector<shape>& inputs) const
    {
        std::vector<shape::dynamic_dimension> referenced_dims;
        bool referenced_dims_founded = false;

        // Find referenced dimensions
        for(const auto& input : inputs)
        {
            if(input.dynamic() && !referenced_dims_founded)
            {
                referenced_dims = input.dyn_dims();
                referenced_dims_founded = true;
                break;
            }
        }

        std::vector<shape> converted_inputs;
        for(const auto& input : inputs)
        {
            if(input.dynamic())
            {
                auto dyn_dims = input.dyn_dims();
                for(std::size_t i = 0; i < dyn_dims.size(); i++)
                {
                    if(i != axis)
                    {
                        // Verify compatibility for non-concatenated axes
                        if(referenced_dims[i] != dyn_dims[i])
                        {
                            MIGRAPHX_THROW("CONCAT: Existed dynamic shapes have incompatible ranges on axis " + 
                            std::to_string(i));
                        }
                    }
                }
                converted_inputs.push_back(input);
            }
            else
            {
                auto lens = input.lens();
                std::vector<shape::dynamic_dimension> dyn_dims;
                for(std::size_t i = 0; i < lens.size(); i++)
                {
                    if(i == axis)
                    {
                        dyn_dims.push_back({lens[i], lens[i]});
                    }
                    else if(referenced_dims_founded)
                    {
                        // Check if static size fits in dynamic range
                        if(lens[i] < referenced_dims[i].min || lens[i] > referenced_dims[i].max)
                        {
                            MIGRAPHX_THROW("CONCAT: Static shape size " + std::to_string(lens[i]) + 
                                " is outside dynamic range [" + 
                                std::to_string(referenced_dims[i].min) + ", " +
                                std::to_string(referenced_dims[i].max) + "] on axis " + 
                                std::to_string(i));
                        }
                        dyn_dims.push_back(referenced_dims[i]);
                    }
                    else
                    {
                        dyn_dims.push_back({lens[i], lens[i]});
                    }
                }
                converted_inputs.push_back({input.type(), dyn_dims});
            }
        }
        inputs = std::move(converted_inputs);
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        // inputs can contain 1 or more shapes (variadic).  compute_shape_op ensures there must
        // be at least 1.
        check_shapes{inputs, *this, true}.same_ndims().same_type();

        // Check if we have mixed static and dynamic shapes
        bool has_static = std::any_of(inputs.begin(), inputs.end(), 
                                    [](const shape& s) { return !s.dynamic(); });
        bool has_dynamic = std::any_of(inputs.begin(), inputs.end(), 
                                    [](const shape& s) { return s.dynamic(); });
        
        if(has_static && has_dynamic)
        {
            handle_mixed_shapes(inputs);
        }

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
            // Calculate the dynamic input shapes for the non-concat axes
            auto common_dyn_dims = compute_common_dyn_dims(inputs);
            
            // Update the dynamic dimensions for the concat axis
            std::size_t new_min = 0;
            std::size_t new_max = 0;
            for(const auto& input : inputs)
            {
                auto ddim = input.dyn_dims()[axis];
                new_min += ddim.min;
                new_max += ddim.max;
            }

            common_dyn_dims[axis] = migraphx::shape::dynamic_dimension{new_min, new_max};
            return {inputs[0].type(), common_dyn_dims};
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
