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
#ifndef MIGRAPHX_GUARD_OPERATORS_DYN_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_DYN_SLICE_HPP

#include <migraphx/algorithm.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Slice operator whose bounds are only known at run time.
///
/// The starts and ends are always supplied as inputs. The attribute of the same name describes
/// that input at compile time with an expression that evaluates to what the input will hold at
/// run time; a constant bound is a sym::lit. The axes have to be known when the shape is
/// computed, so they are an attribute only.
///
/// An end before its start is rejected: at run time by compute(), and when the shape is computed
/// for the bounds that put the end before the start over their whole range. A slice that is only
/// empty (end equal to start) is allowed and produces a zero-length dimension.
///
/// Attributes:
/// axes: axes to slice over
/// starts: slice starting indices
/// ends: slice ending indices
///
/// Parameters:
/// data: the input tensor to slice (static or symbolic shape)
/// starts_input: starting indices of the slice (static shape, 1D)
/// ends_input: ending indices of the slice (static shape, 1D)
struct dyn_slice
{
    std::vector<int64_t> axes{};
    std::vector<sym::expr> starts{};
    std::vector<sym::expr> ends{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.starts, "starts"), f(self.ends, "ends"));
    }

    /// Ensure the axes attribute is within limits, and clip starts and ends to the sliced axis
    /// length. Symbolic bounds are clipped symbolically; see tune_attribute_sym().
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

    std::string name() const { return "dyn_slice"; }

    /// Check the attributes against each other and against the inputs.
    void check_inputs_and_attributes(const std::vector<shape>& inputs) const
    {
        if(axes.empty() or starts.size() != axes.size() or ends.size() != axes.size())
        {
            MIGRAPHX_THROW("DYN_SLICE: axes, starts, and ends attributes must all be set and "
                           "have the same length");
        }
        auto empty_expr = [](const sym::expr& e) { return e.empty(); };
        if(std::any_of(starts.begin(), starts.end(), empty_expr) or
           std::any_of(ends.begin(), ends.end(), empty_expr))
        {
            MIGRAPHX_THROW("DYN_SLICE: starts and ends attributes cannot hold an empty "
                           "expression");
        }
        // The inputs carry the run-time value of the starts and ends attributes, so there is one
        // entry per sliced axis.
        check_shapes{inputs.begin() + 1,
                     inputs.end(),
                     std::string("DYN_SLICE: inputs (starts, ends)"),
                     false}
            .only_dims(1)
            .same_dims();
        if(inputs[1].lens().front() != axes.size())
        {
            MIGRAPHX_THROW(
                "DYN_SLICE: input length (" + migraphx::to_string(inputs[1].lens().front()) +
                ") does not match attribute length (" + migraphx::to_string(axes.size()) + ")");
        }
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(3);
        check_inputs_and_attributes(inputs);
        const auto& input_shape = inputs.front();
        if(input_shape.dynamic() and not input_shape.symbolic())
            MIGRAPHX_THROW("DYN_SLICE: data input must have a static or symbolic shape");

        auto sym_in = input_shape.to_symbolic();
        auto dds    = sym_in.dyn_dims();
        std::vector<sym::expr> extents(axes.size());
        std::transform(ends.begin(),
                       ends.end(),
                       starts.begin(),
                       extents.begin(),
                       [](const auto& end, const auto& start) { return end - start; });
        auto zero = sym::lit(std::int64_t{0});
        migraphx::for_each(
            axes.begin(), axes.end(), extents.begin(), [&](auto axis, const auto& extent) {
                // Negative over its whole range means compute() would reject every run-time
                // value, so there is no point compiling the program.
                if(sym::strict_less(extent, zero).value_or(false))
                    MIGRAPHX_THROW("DYN_SLICE: axis " + migraphx::to_string(axis) +
                                   ": end is always before start, extent " + extent.to_string() +
                                   " is negative over its whole range");
                // An extent that merely might be negative cannot be ruled out by interval
                // arithmetic when the bounds are independent symbols, so clamp at zero to keep
                // the dimension non-negative. The clamp folds away when the subtraction is
                // provably non-negative.
                dds[axis] = shape::dynamic_dimension{sym::fold_max(extent, zero)};
            });
        shape result{input_shape.type(), std::move(dds), sym_in.dyn_strides()};
        // A slice is a view, so a fully concrete result of a static input must stay static.
        if(not input_shape.symbolic() and result.is_fixed())
            return result.to_static();
        return result;
    }

    argument compute(const dyn_output&, std::vector<argument> args) const
    {
        const auto& input = args.front();
        auto input_shape  = input.get_shape();
        auto read         = [](const argument& arg) {
            std::vector<int64_t> result;
            arg.visit([&](auto values) { result = values.template to_vector<int64_t>(); });
            return result;
        };
        auto axes_attrs = this->attributes().at("normalize_axes");
        // Only use the starts_input and ends_input for the output slice. Not the attributes.
        auto norm_starts = normalize_indices(
            read(args[1]), axes, input_shape, axes_attrs.at("starts"), "DYN_SLICE: starts input");
        auto norm_ends = normalize_indices(
            read(args[2]), axes, input_shape, axes_attrs.at("ends"), "DYN_SLICE: ends input");

        // Get end-start for output dimension sizes. Reject if ends before starts (no wrap around).
        std::vector<std::size_t> extents(axes.size());
        std::transform(norm_ends.begin(),
                       norm_ends.end(),
                       norm_starts.begin(),
                       extents.begin(),
                       [](auto end, auto start) {
                           if(end < start)
                               MIGRAPHX_THROW("DYN_SLICE: end (" + migraphx::to_string(end) +
                                              ") is before start (" + migraphx::to_string(start) +
                                              ")");
                           return std::size_t(end - start);
                       });
        auto new_lens = input_shape.lens();
        std::vector<std::size_t> start_indices(input_shape.ndim(), 0);
        migraphx::for_each(axes.begin(),
                           axes.end(),
                           norm_starts.begin(),
                           [&](auto axis, auto start) { start_indices[axis] = start; });
        migraphx::for_each(axes.begin(), axes.end(), extents.begin(), [&](auto axis, auto extent) {
            new_lens[axis] = extent;
        });
        auto offset = input_shape.index(start_indices) * input_shape.type_size();
        shape output_shape{input_shape.type(), new_lens, input_shape.strides()};
        return {output_shape, [=] { return input.data() + offset; }};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
