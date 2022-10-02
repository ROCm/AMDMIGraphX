/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_BROADCAST_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// The broadcast operator performs the numpy-style broadcasting of an axis of a given tensor. This
/// is achieved primarily by setting the stride of the broadcasted axis to zero. Linear indicies are
/// computed from multi-indicies by computing the inner product on the multi-index with the strides.
/// For example, if we have a tensor A(2,3) it has lengths of (2,3) and strides of (3,1). If we want
/// to compute the linear offset that corresponds to the element on the 2nd row (i = 1) and 3rd
/// column (j = 2), we compute the following inner product (1,2) dot (3, 1) = 1*3 + 2*1 = 5. It is
/// obvious from there that we can negate the effects of a given axis by setting the stride of that
/// axis to zero.
struct broadcast
{
    uint64_t axis = 0;
    std::vector<std::size_t> broadcast_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.broadcast_lens, "out_lens"));
    }

    std::string name() const { return "broadcast"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);
        auto s0 = inputs.at(0);
        auto t     = s0.type();

		if (inputs.size() == 1)
		{
			std::vector<size_t> bcast_strides(broadcast_lens.size(), 0);
			// the broadcast op is deprecated now, so not handling the negative
			// value of axis anymore
			if(axis >= broadcast_lens.size())
			{
				MIGRAPHX_THROW("BROADCAST : axis is out of range");
			}

			if(broadcast_lens.size() - axis < s0.lens().size())
			{
				MIGRAPHX_THROW("BROADCAST: (broadcast ndims - axis) is less than s0 ndims");
			}

			if(not std::equal(s0.lens().begin(), s0.lens().end(), broadcast_lens.begin() + axis))
			{
				MIGRAPHX_THROW("BROADCAST: when broadcasting, succeeding sizes must match");
			}
			std::copy(s0.strides().begin(), s0.strides().end(), bcast_strides.begin() + axis);

			shape output{t, broadcast_lens, std::move(bcast_strides)};
			if(output.elements() < s0.elements())
				MIGRAPHX_THROW("BROADCAST: output size must be greater than or equal to s0 size");
			return output;
		}
		else
		{
            if(s0.dynamic() and s1.dynamic()) 
			{
				auto bcast_max_lens = compute_broadcasted_lens(s0.max_lens(), s1.max_lens());
				auto bcast_min_lens = compute_broadcasted_lens(s0.min_lens(), s1.min_lens());
				auto bcast_opt_lens = compute_broadcasted_lens(s0.opt_lens(), s1.opt_lens());

				std::vector<shape::dynamic_dimension> output_dyn_dims = {};
				for(size_t i = 0; i < bcast_max_lens.size(); ++i)
				{
					output_dyn_dims.push_back(shape::dynamic_dimension{
						min_spatial_dims[i], max_spatial_dims[i], opt_spatial_dims[i]});
				}
				return {t, output_dyn_dims};
			}
            else if(not s0.dynamic() and not s1.dynamic())
            {
                auto bcast_lens = compute_broadcasted_lens(s0.lens(), s1.lens());
				std::vector<size_t> bcast_strides(broadcast_lens.size(), 0);
				std::copy(s0.strides().begin(), s0.strides().end(), bcast_strides.begin() + axis);
				return {t, std::move(bcast_lens), std::move(bcast_strides)};
            }
            else
            {
                MIGRAPHX_THROW(
                    "BROADCAST: s0 and s1 are not both dynamic or static");
            }
		}
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return args[0].reshape(output_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
