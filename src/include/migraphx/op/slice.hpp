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
#include <migraphx/op/normalize_attribute.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Slice operator that accepts varible axes, starts and ends.
 * Input arguments:
 * data: the input tensor to slice (dynamic or static shape)
 * starts: starting indicies of slice (optional, static shape)
 * ends: ending indicies of slice (optional, static shape)
 * input_axes: axes to slice over (optional, static shape)
 */
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

    /**
     * Computes the slice output shape dimensions for given starts, ends, and optionally axes.
     * Templated to also handle tensor views.
     * Assumes in_starts and in_ends are normalized; in_axes are valid.
     */
	template <class Inds>
    std::vector<std::size_t> lens_calc(std::vector<std::size_t> lengths, Inds in_starts, Inds in_ends, optional<Inds> in_axes)
    {
        auto new_lens = lengths;
        for(std::size_t i = 0; i < in_axes.size(); ++i)
        {
            auto axis = in_axes[i];
            new_lens[axis] = in_ends[i] - in_starts[i];
        }
        return new_lens;
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 3, 4);
        auto input_shape = inputs[0];
        if(inputs.size == 1)
        {
            auto t           = input_shape.type();
            // TODO:  When support for dynamic shapes is added to normalize_attributes,
            //  remove this restriction.
            if(input_shape.dynamic() and std::any_of(axes.begin(), axes.end(), [&](auto axis) {
                   return not input_shape.dyn_dims()[axis].is_fixed();
               }))
            {
                MIGRAPHX_THROW("SLICE: slicing is not allowed on non-fixed dynamic input axis ");
            }

            // Doesn't handle optimals
            if(input_shape.dynamic())
            {
                old_lens = input_shape.max_lens();
                new_mins = input_shape.min_lens();
            }
            else
            {
                old_lens = input_shape.lens();
                old_strides = input_shape.strides();
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
                return shape{t,
                            lens_calc(input_shape.lens(), starts, ends, axes),
                            input_shape.strides()};
            }
        }
        else
        {
            // check that starts, ends, and optionally input_axes have the same dimension and are static
            check_shapes{inputs.cbegin() + 1, inputs.cend(), "SLICE: inputs (starts, ends, and input_axes)", false}.only_dims(1).same_dims();
            auto dds = input_shape.to_dynamic().dyn_dims();
            if(inputs.size() == 3)
            {
                if(inputs[1].lens().at(0) != axes.size())
                {
                    MIGRAPHX_THROW("SLICE: inputs starts and ends do not have the same length as axes attribute");
                }
                std::for_each(
                    axes.cbegin(),
                    axes.cend(),
                    [&](const auto& axis) { dds.at(axis) = {0, dds.at(axis).max}; }
                );
            }
            else
            {
                // if axes is an input, then all the output dimensions could be 0 to the max value
                std::transform(
                    dds.begin(),
                    dds.end(),
                    dds.begin(),
                    [](auto dd) { return {0, dd.max}; }
                );
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
        return offset;
    }

    /**
	 * Calculates the starting offset for the sliced tensor.
	 * Used in compute when starts and optionally input_axes are variable.
     *
	 * \param s static input shape
	 * \param input_starts starting indices of slice
	 * \param input_axes optional axes to slice on, if not present will use the axes attribute.
	 */
	template <class IndView>
    auto compute_offset(const shape& s,
						IndView input_starts,
						optional<IndView> input_axes) const
    {
		auto calc_offset = [&](const auto& ax_vec) {
			auto ret = 0;
            for(std::size_t i = 0; i < ax_vec.size(); ++i)
            {
                auto axis = ax_vec[i];
                offset += input_starts[i] * s.strides[axis];
            }
			return ret;
		};
		return input_axes ? calc_offset(input_axes.value()) : calc_offset(axes);
    }

	/**
	 * Clamps the input_starts and input_ends to the input tensor dimensions and makes them positive indicies.
	 * Checks that the input_axes are valid.
	 */
	template <class Inds>
	std::unordered_map<Inds> normalize_inputs(
            shape input_shape,
            Inds input_starts,
            Inds input_ends,
            optional<Inds> input_axes) const
	{
        auto normalize_index = [&](auto indices, auto in_axes){
            std::transform(
                in_axes.begin(),
                in_axes.end(),
                indices.begin(),
                indices.end(),
                indices.begin(),
                [](auto axis, auto index){
                    auto dim = input_shape.lens().at(axis);
                    if(index < 0)
                    {
                        index += dim;
                    }
                    if(index < 0)
                    {
                        index = 0;
                    }
                    else if(index > dim)
                    {
                        index = dim;
                    }
                    return index;
            });
            return indices;
        };
		if(not input_axes)
		{
            return {
                "input_starts":normalize_index(input_starts, input_axes),
                "input_ends":normalize_index(input_ends, input_axes)
            };
		}
		else
		{
            // normalize axes
            auto shape_ndim = shape.ndim();
            std::transform(axes.begin(), axes.end(), axes.begin(), [&](auto axis){
                if(axis < 0)
                {
                    axis += shape_ndim;
                }
                if(axis < 0 or axis >= shape_ndim)
                {
                    MIGRAPHX_THROW("SLICE: entry of input_axes out of bounds");
                }
                return axis;
            });
            return {
                "input_starts":normalize_index(input_starts),
                "input_ends":normalize_index(input_ends),
                "input_axes":axes
            };
        }
	}

    argument compute(const shape&, std::vector<argument> args) const
    {
        auto input_shape = args[0].get_shape();
        
        std::size_t offset;
        switch(args.size())
        {
            case 1:
                offset = compute_offset(input_shape) * output_shape.type_size();
                return {normalize_compute_shape({input_shape}), [=] { return input.data() + offset; }};
            case 3:
                shape output_shape;
                visit_all(args[1], args[2])([&](auto input_starts, auto input_ends){
                    auto norm_inputs = normalize_inputs(input_shape, input_starts, input_ends);
                    offset = compute_offset(input_shape, norm_inputs.at("input_starts"));
                    output_shape = {
                        lens_calc(
                            input_shape.lens(),
                            norm_inputs.at("input_starts"),
                            norm_inputs.at("input_ends")
                        ),
                        input_shape.strides()
                    };
                });
                return {output_shape, [=] { return input.data() + offset; }};
            case 4:
                shape output_shape;
                visit_all(args[1], args[2], args[3])([&](auto input_starts, auto input_ends, auto axes){
                    auto norm_inputs = normalize_inputs(input_shape, input_starts, input_ends);
                    offset = compute_offset(input_shape, norm_inputs.at("input_starts"), norm_inputs.at("input_axes"));
                    output_shape = {
                        lens_calc(
                            input_shape,
                            norm_inputs.at("input_starts"),
                            norm_inputs.at("input_ends"),
                            norm_inputs.at("input_axes")
                        ),
                        input_shape.strides()
                    };
                });
                return {output_shape, [=] { return input.data() + offset; }};
            default:
                MIGRAPHX_THROW("SLICE: invalid number of inputs");
        }
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
