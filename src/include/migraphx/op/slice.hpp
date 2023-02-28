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

    /**
     * Ensure that attribute vectors axes, starts, and ends are all the same size and values are in
     * limits.
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
        return offset;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto input_shape = inputs[0];
        auto t           = input_shape.type();

        // TODO:  When support for dynamic shapes is added to normalize_attributes,
        //  remove this restriction.
        if(input_shape.dynamic() and std::any_of(axes.begin(), axes.end(), [&](auto axis) {
               return not input_shape.dyn_dims()[axis].is_fixed();
           }))
        {
            MIGRAPHX_THROW("SLICE: slicing is not allowed on non-fixed dynamic input axis ");
        }

        // For a static shape, old_lens will be adjusted to a new size
        // for those axes that are sliced.
        // For dynamic shape, the adjusted old_lens become the new max values,
        // while updating the old mins and opts if possible.
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
            // For static shape (including during eval step after a dynamic input) the strides are
            // indexed into the pre-slice array, so they are larger than the apparent size of the
            // resulting shape.
            old_strides = input_shape.strides();
        }

        std::vector<std::size_t> new_lens = old_lens;
        for(std::size_t i = 0; i < axes.size(); i++)
        {
            auto axis            = axes[i];
            size_t sliced_length = ends[i] - starts[i];
            // A Numpy indexing convention: a slice size larger than the actual dimension
            // is legal and the "ends" value is clipped to the axis size
            new_lens[axis] = std::min(new_lens[axis], sliced_length);
            if(input_shape.dynamic())
            {
                // TODO: when non-fixed shape slicing is allowed, this will be different than
                // sliced_length, making use of TBD start/end values.
                std::size_t sliced_min_length = ends[i] - starts[i];
                // if the slice size is smaller than maxes but larger than mins
                new_mins[axis] = std::min(sliced_min_length, new_mins[axis]);

                auto sliced_opt_length = ends[i] - starts[i];
                if(new_opts[axis] != 0)
                    new_opts[axis] = sliced_opt_length;
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
