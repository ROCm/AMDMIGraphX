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
#include <migraphx/dyn_output.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct slice
{
    std::vector<int64_t> axes;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.starts, "starts"), f(self.ends, "ends"));
    }

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

    // Convert negative index to a value counting backwards from the max index, per numpy slicing
    // convention, and clip large values to 1 more than the largest index.
    auto fix_index(const std::vector<std::size_t>& lens, std::size_t axis, int64_t index) const
    {
        int64_t r = std::min(index, static_cast<int64_t>(lens[axis]));
        if(r < 0)
            r += lens[axis];
        return std::size_t(r);
    }

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
                offset += fix_index(lens, axis, starts[i]) * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += fix_index(lens, axis, starts[axis]) * strides[axis];
            }
        }
        return offset;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto input_shape = inputs[0];
        auto t           = input_shape.type();
        // For a static shape, old_lens will be adjusted to a new size
        // for those axes that are sliced.
        // For dynamic shape, the adjusted old_lens become the new max values,
        // while retaining the old mins and opts if possible.
        std::vector<std::size_t> new_mins;
        std::vector<std::size_t> new_opts;
        std::vector<std::size_t> old_lens;
        std::vector<std::size_t> old_strides;
        if(input_shape.dynamic())
        {
            old_lens = input_shape.max_lens();
            new_mins = input_shape.min_lens();
            new_opts = input_shape.opt_lens();
        }
        else
        {
            old_lens = input_shape.lens();
            // For static shape (including eval after a dynamic input) the strides are indexed into
            // the pre-slice array, so they are larger than the apparent size of the resulting
            // shape.
            old_strides = input_shape.strides();
        }

        if(std::any_of(
               axes.begin(), axes.end(), [&](auto i) { return (i >= old_lens.size() or i < 0); }))
        {
            MIGRAPHX_THROW("SLICE: input axis " + to_string_range(axes) + " out of range");
        }

        if(starts.size() != axes.size() or axes.size() != ends.size())
        {
            MIGRAPHX_THROW(
                "SLICE: inputs \"starts\", \"ends\", and \"axes\" must all be the same size");
        }

        std::vector<std::size_t> new_lens = old_lens;
        for(std::size_t i = 0; i < axes.size(); i++)
        {
            auto axis = axes[i];
            auto sliced_length =
                fix_index(old_lens, axis, ends[i]) - fix_index(old_lens, axis, starts[i]);
            // A Numpy indexing convention: a slice size larger than the actual dimension
            // is legal and the "ends" value is clipped to the axis size
            new_lens[axis] = std::min(new_lens[axis], sliced_length);
            if(input_shape.dynamic())
            {
                auto sliced_min_length = fix_index(new_mins, axis, ends[i]) - fix_index(new_mins, axis, starts[i]);
                new_mins[axis] = std::min(sliced_min_length, new_mins[axis]);
                if(new_opts[axis] < new_mins[axis] or new_opts[axis] > new_lens[axis])
                    new_opts[axis] = 0;
            }
        }
        if(input_shape.dynamic())
        {
            return shape{t, new_mins, new_lens, new_opts};
        }
        else
        {
            return shape{t, new_lens, old_strides};
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        auto input = args[0];

        auto offset = compute_offset(input.get_shape()) * dyn_out.computed_shape.type_size();
        return {dyn_out.computed_shape, [=] { return input.data() + offset; }};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
