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
#ifndef MIGRAPHX_GUARD_OPERATORS_SLICE_DYNAMIC_HPP
#define MIGRAPHX_GUARD_OPERATORS_SLICE_DYNAMIC_HPP

#include <migraphx/check_shapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Version of the slice operator that has variable (input defined rather than compile-time constant)
 * starts and ends with optionally varible axes.
 * Either the axes attribute must be set or it has to be an input.
 * Empty axes attribute and no axes input is not valid.
 *
 * Input arguments:
 * data the input tensor to slice (dynamic or static shape)
 * starts starting indicies of slice (static shape)
 * ends ending indicies of slice (static shape)
 * input_axes optional axes to slice over (static shape)
 */
struct slice_dynamic
{
	std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    value attributes() const
    {
        value normalize     = value::object{};
        normalize["axes"]   = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "slice_dynamic"; }


	shape normalize_compute_shape(std::vector<shape> inputs) const
		check_shapes{inputs, *this, true}.has(3, 4);
		// check that starts, ends, and optionally input_axes have the same dimension and are static
		check_shapes{inputs.cbegin() + 1, inputs.cend()}, "SLICE_DYNAMIC inputs (starts, ends, and input_axes)", false}.only_dims(1).same_dims();
		auto data_shape = inputs[0];
		auto dds = data_shape.to_dynamic().dyn_dims();
		if(inputs.size() == 4)
		{
			// if axes is an input, then all the output dimensions could be 0 to the max value
			std::transform(
				dds.begin(),
				dds.end(),
				dds.begin(),
				[](auto dd) { return {0, dd.max}; }
			);
		}
		else
		{
			if(inputs[1].lens().at(0) != axes.size())
			{
				MIGRAPHX_THROW("SLICE_DYNAMIC: inputs starts and ends do not have the same length as axes attribute");
			}
			std::for_each(
				axes.cbegin(),
				axes.cend(),
				[&](const auto& axis) { dds.at(axis) = {0, dds.at(axis).max}; }
			);
		}
		return shape{data_shape.type(), dds};
	}
	
	/**
	 * Clamps the starts and ends to the input tensor dimensions and makes them positive indicies.
	 * Checks that the input_axes are valid.
	 */
	template <class Index>
	auto normalize_inputs(const std::vector<Index>& starts,
						  const std::vector<Index>& ends,
						  optional<std::reference_wrapper<std::vector<Index>>> input_axes) const
	{
		if(input_axes)
		{
			auto out_axes = input_axes;
			std::transform(input_axes.cbegin(),
						   input_axes.cend(),
						   out_axes.begin(),
						   [](
		}
		else
		{

		}
	}

	/**
	 * Calculates the starting offset for the sliced tensor.
	 * Used during the the compute().
	 *
	 * \param s static input shape
	 * \param starts starting indices of slice
	 * \param input_axes optional axes to slice on, if not present will use the axes attribute.
	 */
	template <class IndView>
    auto compute_offset(const shape& s,
						IndView starts,
						optional<IndView> input_axes) const
    {
		auto calc_offset = [&](const auto& ax_vec) {
			auto ret = 0;
            for(std::size_t i = 0; i < ax_vec.size(); ++i)
            {
                auto axis = ax_vec[i];
                offset += starts[i] * s.strides[axis];
            }
			return ret;
		};
		return input_axes ? calc_offset(input_axes.value()) : calc_offset(axes);
    }

    argument compute(const shape& dyn_shape, std::vector<argument> args) const
    {
		// compute static shape using the inputs
		shape output_shape;
		if(args.size() == 3)
		{
			visit_all(args[1], args[2])([&](auto starts, auto ends){
				
			});
		}
		else
		{
			visit_all(args[1], args[2], args[3])([&](auto starts, auto ends, auto input_axes){

			});
		}

		// then do the offset calculation and aliasing
		// Slice works by setting the starting offset and then using the original input strides with the sliced dimensions
	}

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
