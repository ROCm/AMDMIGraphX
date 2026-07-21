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
#ifndef MIGRAPHX_GUARD_OPERATORS_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SLICE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/enum.hpp>
#include <array>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Slice operator that accepts variable axes, starts and ends.
/// All of `starts`, `ends`, and `axes` attributes must be supplied.
///
/// `mode` specifies what the inputs to slice are:
/// one_input: slice(input);
/// starts_input: slice(input, starts);
/// ends_input: slice(input, ends);
/// axes_input: slice(input, axes);
/// starts_ends_input: slice(input, starts, ends);
/// starts_axes_input: slice(input, starts, axes);
/// ends_axes_input: slice(input, ends, axes);
/// starts_ends_axes_input: slice(input, start, ends, axes);
///
/// Attributes:
/// axes: axes to slice over
/// starts: slice starting indices
/// ends: slice ending indices
///
/// Parameters:
/// data: the input tensor to slice (dynamic or static shape)
/// starts_input: starting indices of slice (optional, static shape)
/// ends_input: ending indices of slice (optional, static shape)
/// axes_input: axes to slice over (optional, static shape)
struct slice
{
    MIGRAPHX_NESTED_ENUM_CLASS(slice_mode,
                               one_input,
                               starts_input,
                               ends_input,
                               axes_input,
                               starts_ends_input,
                               starts_axes_input,
                               ends_axes_input,
                               starts_ends_axes_input);

    std::vector<int64_t> axes{};
    std::vector<dim_like> starts{};
    std::vector<dim_like> ends{};
    slice_mode mode = slice_mode::one_input;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"),
                    f(self.starts, "starts"),
                    f(self.ends, "ends"),
                    f(self.mode, "mode"));
    }

    /// Ensure that attribute axes is within limits.
    /// Will attempt to normalize starts and ends; but will use the dynamic_dimension.max
    /// values for dynamic shapes. This makes it so you have to renormalize for
    /// non-fixed dynamic_dimensions.
    value attributes() const
    {
        value normalize_axes     = value::object{};
        normalize_axes["axes"]   = value::array{normalize_attribute::include_min};
        normalize_axes["starts"] = value::array{normalize_attribute::clip_max,
                                                normalize_attribute::clip_min,
                                                normalize_attribute::include_max,
                                                normalize_attribute::use_len,
                                                normalize_attribute::include_min};
        normalize_axes["ends"]   = value::array{normalize_attribute::clip_max,
                                              normalize_attribute::clip_min,
                                              normalize_attribute::include_max,
                                              normalize_attribute::use_len,
                                              normalize_attribute::include_min};
        return {{"normalize_axes", normalize_axes}, {"fillcolor", "#FFA500" /* orange */}};
    }

    std::string name() const { return "slice"; }

    /// Computes the slice output shape dimensions for given starts, ends,and axes.
    /// Templated to also handle tensor views.
    /// Possibly different type between [in_starts, in_ends] and [in_axes] if in_axes is this
    /// object's axes attribute. Assumes in_starts and in_ends are normalized; in_axes are valid.
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

    /// Check that the inputs, attributes, and mode are valid.
    void check_inputs_and_attributes(std::vector<shape> inputs) const
    {
        auto input_shape = inputs[0];
        if(axes.size() != starts.size() or starts.size() != ends.size())
            MIGRAPHX_THROW("SLICE: Invalid attributes configuration. Not the same number of "
                           "dimensions. axes: " +
                           migraphx::to_string(axes.size()) +
                           " starts: " + migraphx::to_string(starts.size()) +
                           " ends: " + migraphx::to_string(ends.size()));

        if(inputs.size() == 1)
        {
            if(any_sym(starts) or any_sym(ends))
                MIGRAPHX_THROW("SLICE: Invalid attributes: symbolic in attribute for 1 input slice");
            if(mode != slice_mode::one_input)
                MIGRAPHX_THROW("SLICE: Invalid mode for 1 input");
            return;
        }
        // check that inputs [1, end) are all 1D, have the same dimension, and are static
        check_shapes{inputs.begin() + 1,
                     inputs.end(),
                     std::string("SLICE: inputs (starts_input, ends_input, axes_input)"),
                     false}
            .only_dims(1)
            .same_dims();
        if(inputs.at(1).lens().at(0) != axes.size())
        {
            MIGRAPHX_THROW("SLICE: varable input and attributes mismatch: input[1] length (" +
                           to_string(inputs[1].lens().at(0)) +
                           ") != attribute number of dimensions (" + to_string(axes.size()) + ")");
        }
        if(inputs.size() == 2)
        {
            std::vector<slice_mode> two_input_modes = {
                slice_mode::starts_input, slice_mode::ends_input, slice_mode::axes_input};
            if(not contains(two_input_modes, mode))
            {
                MIGRAPHX_THROW("SLICE: Invalid mode for 2 inputs");
            }
        }
        else if(inputs.size() == 3)
        {
            std::vector<slice_mode> three_input_modes = {slice_mode::starts_ends_input,
                                                         slice_mode::starts_axes_input,
                                                         slice_mode::ends_axes_input};
            if(not contains(three_input_modes, mode))
            {
                MIGRAPHX_THROW("SLICE: Invalid mode for 3 inputs");
            }
        }
        else
        {
            if(mode != slice_mode::starts_ends_axes_input)
            {
                MIGRAPHX_THROW("SLICE: Invalid mode for 4 inputs");
            }
        }
        return;
    }

    // Static and symbolic inputs share this path; the result is demoted back to
    // static when fully fixed (slice is a view).
    shape symbolic_compute_shape(const shape& s) const
    {
        assert(starts.size() == axes.size() and ends.size() == axes.size());
        auto sym_in      = s.to_symbolic();
        auto dds         = sym_in.dyn_dims();
        auto start_exprs = to_sym_exprs(starts);
        auto end_exprs   = to_sym_exprs(ends);
        for(std::size_t i = 0; i < axes.size(); ++i)
            dds[axes[i]] = shape::dynamic_dimension{end_exprs[i] - start_exprs[i]};
        shape result{s.type(), std::move(dds), sym_in.dyn_strides()};
        if(not s.symbolic() and result.is_fixed())
            return result.to_static();
        return result;
    }

    // uses the normalize_axes flag to normalize axes, starts, and ends
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2, 3, 4);
        check_inputs_and_attributes(inputs);
        auto input_shape = inputs[0];
        // fallback for range-based dynamic shapes. Only handling 1 arg case.
        if(input_shape.dynamic() and not input_shape.symbolic())
        {
            if(inputs.size() != 1)
            {
                MIGRAPHX_THROW(
                    "SLICE: range-based dynamic input shapes with variable inputs unsupported.");
            }
            // Non-fixed sliced axis: bounds aren't normalized (can be negative or
            // out-of-bounds), so use a relaxed [0, max] bound. (#5015)
            if(std::any_of(axes.begin(), axes.end(), [&](auto axis) {
                   return not input_shape.dyn_dims()[axis].is_fixed();
               }))
            {
                auto dds = input_shape.dyn_dims();
                for(auto axis : axes)
                    dds[axis] = {0, dds[axis].get_interval().max};
                return shape{input_shape.type(), dds};
            }

            auto new_lens = lens_calc(input_shape.max_lens(), to_ints(starts), to_ints(ends), axes);
            auto dds      = input_shape.dyn_dims();
            for(auto axis : axes)
                dds[axis] = shape::dynamic_dimension{new_lens[axis], new_lens[axis]};
            return shape{input_shape.type(), dds};
        }

        return symbolic_compute_shape(input_shape);
    }

    /// Calculates the starting offset for the sliced tensor.
    /// Used in compute when only data input and all other information are in the attributes.
    ///
    /// s: static input shape
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
                offset += std::get<int64_t>(starts[i]) * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += std::get<int64_t>(starts[axis]) * strides[axis];
            }
        }
        return offset * s.type_size();
    }

    /// Calculates the starting offset for the sliced tensor (for aliasing).
    /// Used for 2-4 inputs to `slice.
    ///
    /// s: static input shape
    /// starts_input: starting indices of slice
    /// ax_vec: axes to slice on
    template <class T>
    auto compute_offset(const shape& s, const T& starts_input, const T& ax_vec) const
    {
        auto ret = 0;
        for(std::size_t i = 0; i < ax_vec.size(); ++i)
        {
            auto axis = ax_vec[i];
            ret += starts_input[i] * s.strides().at(axis);
        }
        return ret * s.type_size();
    }

    /// Used to normalize starts/ends/axes at runtime.
    /// If given, normalize the starts/ends/axes inputs. Otherwise get from operator attributes.
    /// If starts_input or ends_input is not given, assuming starts/ends attributes are only
    /// integers. If starts/ends have symbolics, they should go through the starts_input and
    /// ends_input instead. `axes` attribute should always be correctly normalized at compile-time
    /// because shapes with dynamic rank are not supported.
    ///
    /// input_shape: static shape of the input
    /// starts_input: optional
    /// ends_input: optional
    /// axes_input: optional
    std::unordered_map<std::string, std::vector<int64_t>>
    normalize_starts_ends_axes(shape input_shape,
                               const optional<std::vector<int64_t>>& starts_input,
                               const optional<std::vector<int64_t>>& ends_input,
                               const optional<std::vector<int64_t>>& axes_input) const
    {
        assert(not input_shape.dynamic());
        auto axes_attrs = this->attributes().at("normalize_axes");
        std::vector<int64_t> norm_starts;
        std::vector<int64_t> norm_ends;
        std::vector<int64_t> norm_axes;
        if(axes_input)
        {
            norm_axes = normalize_axes(axes_input.value(),
                                       input_shape,
                                       axes_attrs.at("axes"),
                                       "Slice variable axes_input");
        }
        else
        {
            norm_axes = this->axes;
        }
        if(starts_input)
        {
            norm_starts = normalize_indices(starts_input.value(),
                                            norm_axes,
                                            input_shape,
                                            axes_attrs.at("starts"),
                                            "Slice variable starts_input");
        }
        else
        {
            norm_starts = to_ints(this->starts);
        }
        if(ends_input)
        {
            norm_ends = normalize_indices(ends_input.value(),
                                          norm_axes,
                                          input_shape,
                                          axes_attrs.at("ends"),
                                          "Slice variable input ends");
        }
        else
        {
            norm_ends = to_ints(this->ends);
        }
        return {{"norm_starts", norm_starts}, {"norm_ends", norm_ends}, {"norm_axes", norm_axes}};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        auto input       = args[0];
        auto input_shape = input.get_shape();
        if(args.size() == 1)
        {
            std::size_t offset = compute_offset(input_shape);
            return {dyn_out.computed_shape, [=] { return input.data() + offset; }};
        }
        else
        {
            std::unordered_map<std::string, std::vector<int64_t>> norm_inputs;
            if(mode == slice_mode::starts_input)
            {
                args[1].visit([&](auto starts_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   starts_input.template to_vector<int64_t>(),
                                                   this->ends,
                                                   this->axes);
                });
            }
            else if(mode == slice_mode::ends_input)
            {
                args[1].visit([&](auto ends_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   this->starts,
                                                   ends_input.template to_vector<int64_t>(),
                                                   this->axes);
                });
            }
            else if(mode == slice_mode::axes_input)
            {
                args[1].visit([&](auto axes_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   this->starts,
                                                   this->ends,
                                                   axes_input.template to_vector<int64_t>());
                });
            }
            else if(mode == slice_mode::starts_ends_input)
            {
                // attr axes set; inputs are (data, starts_input, ends_input)
                visit_all(args[1], args[2])([&](auto starts_input, auto ends_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   starts_input.template to_vector<int64_t>(),
                                                   ends_input.template to_vector<int64_t>(),
                                                   this->axes);
                });
            }
            else if(mode == slice_mode::starts_axes_input)
            {
                visit_all(args[1], args[2])([&](auto starts_input, auto axes_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   starts_input.template to_vector<int64_t>(),
                                                   this->ends,
                                                   axes_input.template to_vector<int64_t>());
                });
            }
            else if(mode == slice_mode::ends_axes_input)
            {
                visit_all(args[1], args[2])([&](auto ends_input, auto axes_input) {
                    norm_inputs =
                        normalize_starts_ends_axes(input_shape,
                                                   this->starts,
                                                   ends_input.template to_vector<int64_t>(),
                                                   axes_input.template to_vector<int64_t>());
                });
            }
            else // mode == slice_mode::starts_ends_axes_input
            {
                visit_all(args[1], args[2], args[3])(
                    [&](auto starts_input, auto ends_input, auto axes_input) {
                        norm_inputs =
                            normalize_starts_ends_axes(input_shape,
                                                       starts_input.template to_vector<int64_t>(),
                                                       ends_input.template to_vector<int64_t>(),
                                                       axes_input.template to_vector<int64_t>());
                    });
            }
            auto offset = compute_offset(
                input_shape, norm_inputs.at("norm_starts"), norm_inputs.at("norm_axes"));
            shape calc_shape = shape{input_shape.type(),
                                     lens_calc(input_shape.lens(),
                                               norm_inputs.at("norm_starts"),
                                               norm_inputs.at("norm_ends"),
                                               norm_inputs.at("norm_axes")),
                                     input_shape.strides()};
            return {calc_shape, [=] { return input.data() + offset; }};
        }
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
