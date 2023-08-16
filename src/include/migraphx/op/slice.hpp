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
#ifndef MIGRAPHX_GUARD_OPERATORS_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SLICE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/normalize_attributes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Slice operator that accepts variable axes, starts and ends.
 *
 * Attributes:
 * axes: constant axes to slice over (optional)
 * starts: constant slice starting indices (optional)
 * ends: constant slice ending indices (optional)
 *
 * Parameters:
 * data: the input tensor to slice (dynamic or static shape)
 * input_starts: starting indicies of slice (optional, static shape)
 * input_ends: ending indicies of slice (optional, static shape)
 * input_axes: axes to slice over (optional, static shape)
 */
struct slice
{
    std::vector<int64_t> axes{};
    std::vector<int64_t> starts{};
    std::vector<int64_t> ends{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.starts, "starts"), f(self.ends, "ends"));
    }

    /**
     * Ensure that attribute vectors axes, starts, and ends are all the same size and values are
     * within limits.
     */
    value attributes() const
    {
        value normalize     = value::object{};
        normalize["axes"]   = value::array{normalize_attribute::include_min};
        normalize["starts"] = value::array{normalize_attribute::clip_max,
                                           normalize_attribute::clip_min,
                                           normalize_attribute::include_max,
                                           normalize_attribute::use_len,
                                           normalize_attribute::include_min};
        normalize["ends"]   = value::array{normalize_attribute::clip_max,
                                         normalize_attribute::clip_min,
                                         normalize_attribute::include_max,
                                         normalize_attribute::use_len,
                                         normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "slice"; }

    /**
     * Computes the slice output shape dimensions for given starts, ends,and axes.
     * Templated to also handle tensor views.
     * Possibily different type between [in_starts, in_ends] and [in_axes] if in_axes is this
     * object's axes attribute. Assumes in_starts and in_ends are normalized; in_axes are valid.
     */
    template <class A, class B>
    std::vector<std::size_t>
    lens_calc(const std::vector<std::size_t>& lengths, A in_starts, A in_ends, B in_axes) const
    {
        auto new_lens = lengths;
        for(std::size_t i = 0; i < in_axes.size(); ++i)
        {
            auto axis      = in_axes[i];
            new_lens[axis] = in_ends[i] - in_starts[i];
        }
        return new_lens;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 3, 4);
        auto input_shape = inputs[0];
        if(inputs.size() == 1)
        {
            auto t = input_shape.type();
            if(input_shape.dynamic() and std::any_of(axes.begin(), axes.end(), [&](auto axis) {
                   return not input_shape.dyn_dims()[axis].is_fixed();
               }))
            {
                MIGRAPHX_THROW("SLICE: slicing is not allowed on non-fixed dynamic input axis ");
            }
            if(input_shape.dynamic())
            {
                return shape{t,
                             lens_calc(input_shape.min_lens(), starts, ends, axes),
                             lens_calc(input_shape.max_lens(), starts, ends, axes),
                             {}};
            }
            else
            {
                return shape{
                    t, lens_calc(input_shape.lens(), starts, ends, axes), input_shape.strides()};
            }
        }
        else
        {
            // check that starts, ends, and optionally input_axes are all 1D, have the same
            // dimension, and are static
            check_shapes{inputs.begin() + 1,
                         inputs.end(),
                         std::string("SLICE: inputs (starts, ends, and input_axes)"),
                         false}
                .only_dims(1)
                .same_dims();
            auto dds = input_shape.to_dynamic().dyn_dims();
            if(inputs.size() == 3)
            {
                if(inputs[1].lens().at(0) != axes.size())
                {
                    MIGRAPHX_THROW("SLICE: inputs starts and ends do not have the same dimension "
                                   "as the axes attribute");
                }
                std::for_each(axes.cbegin(), axes.cend(), [&](const auto& axis) {
                    dds.at(axis) = {0, dds.at(axis).max};
                });
            }
            else
            {
                // if axes is an input, then all the output dimensions could be 0 to the max value
                std::transform(dds.begin(), dds.end(), dds.begin(), [](auto dd) {
                    return shape::dynamic_dimension{0, dd.max};
                });
            }
            return shape{input_shape.type(), dds};
        }
    }

    /**
     * Calculates the starting offset for the sliced tensor.
     * Used in compute when only data input and all other information are in the attributes.
     *
     * \param s static input shape
     */
    auto compute_offset(const shape& s) const
    {
        const std::vector<std::size_t>& lens    = s.lens();
        const std::vector<std::size_t>& strides = s.strides();
        auto offset                             = 0;
        if(not axes.empty())
        {
            for(std::size_t i = 0; i < axes.size(); i++)
            {
                auto axis = axes[i];
                offset += starts[i] * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += starts[axis] * strides[axis];
            }
        }
        return offset * s.type_size();
    }

    /**
     * Calculates the starting offset for the sliced tensor (for aliasing).
     * Used when the starts and/or the axes are inputs.
     *
     * \param s static input shape
     * \param input_starts starting indices of slice
     * \param ax_vec axes to slice on
     */
    template <class IndView, class Axes>
    auto compute_offset(const shape& s, const IndView& input_starts, const Axes& ax_vec) const
    {
        auto ret = 0;
        for(std::size_t i = 0; i < ax_vec.size(); ++i)
        {
            auto axis = ax_vec[i];
            ret += input_starts[i] * s.strides().at(axis);
        }
        return ret * s.type_size();
    }

    std::unordered_map<std::string, std::vector<int64_t>>
    normalize_inputs(const shape& input_shape,
                     const std::vector<int64_t>& input_starts,
                     const std::vector<int64_t>& input_ends) const
    {
        auto attrs = this->attributes().at("normalize_axes");
        return {{"input_starts",
                 normalize_indices(input_starts,
                                   this->axes,
                                   input_shape,
                                   attrs.at("starts"),
                                   "Slice variable input_starts")},
                {"input_ends",
                 normalize_indices(input_ends,
                                   this->axes,
                                   input_shape,
                                   attrs.at("ends"),
                                   "Slice variable input_ends")}};
    }

    /**
     * Three input version of the normalize_inputs.
     * This one also checks that the input_axes are valid.
     */
    std::unordered_map<std::string, std::vector<int64_t>>
    normalize_inputs(shape input_shape,
                     const std::vector<int64_t>& input_starts,
                     const std::vector<int64_t>& input_ends,
                     const std::vector<int64_t>& input_axes) const
    {
        auto attrs = this->attributes().at("normalize_axes");
        auto norm_axes =
            normalize_axes(input_axes, input_shape, attrs.at("axes"), "Slice variable input_axes");
        return {{"input_starts",
                 normalize_indices(input_starts,
                                   norm_axes,
                                   input_shape,
                                   attrs.at("starts"),
                                   "Slice variable input_starts")},
                {"input_ends",
                 normalize_indices(input_ends,
                                   norm_axes,
                                   input_shape,
                                   attrs.at("ends"),
                                   "Slice variable input ends")},
                {"input_axes", norm_axes}};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        auto input       = args[0];
        auto input_shape = input.get_shape();
        switch(args.size())
        {
        case 1: {
            std::size_t offset = compute_offset(input_shape);
            return {dyn_out.computed_shape, [=] { return input.data() + offset; }};
        }
        case 3: {
            shape calc_shape;
            std::size_t offset = 0;
            visit_all(args[1], args[2])([&](auto input_starts, auto input_ends) {
                auto norm_inputs = normalize_inputs(input_shape,
                                                    input_starts.template to_vector<int64_t>(),
                                                    input_ends.template to_vector<int64_t>());
                offset = compute_offset(input_shape, norm_inputs.at("input_starts"), this->axes);
                calc_shape = {input_shape.type(),
                              lens_calc(input_shape.lens(),
                                        norm_inputs.at("input_starts"),
                                        norm_inputs.at("input_ends"),
                                        this->axes),
                              input_shape.strides()};
            });
            return {calc_shape, [=] { return input.data() + offset; }};
        }
        case 4: {
            shape calc_shape;
            std::size_t offset = 0;
            visit_all(args[1], args[2], args[3])(
                [&](auto input_starts, auto input_ends, auto input_axes) {
                    auto norm_inputs = normalize_inputs(input_shape,
                                                        input_starts.template to_vector<int64_t>(),
                                                        input_ends.template to_vector<int64_t>(),
                                                        input_axes.template to_vector<int64_t>());
                    offset           = compute_offset(
                        input_shape, norm_inputs.at("input_starts"), norm_inputs.at("input_axes"));
                    calc_shape = shape{input_shape.type(),
                                       lens_calc(input_shape.lens(),
                                                 norm_inputs.at("input_starts"),
                                                 norm_inputs.at("input_ends"),
                                                 norm_inputs.at("input_axes")),
                                       input_shape.strides()};
                });
            return {calc_shape, [=] { return input.data() + offset; }};
        }
        default: {
            // Should never get here; covering in case some code change occurs
            MIGRAPHX_THROW("SLICE: invalid number of inputs");
        }
        }
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
