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
#ifndef MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/common.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct multibroadcast
{
    std::vector<std::size_t> output_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "out_lens"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);

        auto t           = inputs.at(0).type();
        auto s0 = inputs.at(0);

			
		if(s0.lens().empty())
		{
			MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should be > 0");
		}

		auto make_bcast_strides = [&](std::size_t out_num_dims, std::size_t offset)
		{
			std::vector<size_t> bcast_strides(out_num_dims, 0);
			for(std::ptrdiff_t i = s0.lens().size() - 1; i >= 0; i--)
			{
				if(output_lens[i + offset] == s0.lens()[i])
				{
					bcast_strides[i + offset] = s0.strides()[i];
				}
			}
			return bcast_strides;
		};

        if(inputs.size() == 1)
        {
            if(s0.lens().size() > output_lens.size())
            {
                MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should <= output size");
            }

            auto offset = output_lens.size() - s0.lens().size();
            for(std::ptrdiff_t i = s0.lens().size() - 1; i >= 0; i--)
            {
                if(output_lens[i + offset] != s0.lens()[i] and s0.lens()[i] != 1)
                {
                    MIGRAPHX_THROW(
                        "MULTIBROADCAST: input shape {" + to_string_range(s0.lens()) +
                        "} cannot be broadcasted to {" + to_string_range(output_lens) + "}!");
                }
            }

			auto bcast_strides = make_bcast_strides(output_lens.size(), offset);
            return {t, output_lens, std::move(bcast_strides)};
        }
        else
        {
            // need both shapes when handling dynamic case
            // shapes can be dynamic (at compile-time) or static (at evaluation time)
            // this function will be called through compute_output_shape conversion to dyn_output
            // new compute_broadcasted_lens for dynamic shapes
			// do we want this to work in both broadcast directions?
			// s0 and s1 as shape inputs
			// always s0 -> s1 shape or allow s0 to retain the same shape?
			// presuming that it's always s0 -> s1 shape, since that's closer to the current behavior
			// compute_broadcasted_lens() will swap the shapes if s1.size() < s0.size(), may need to make another function
            auto s1 = inputs.at(1);
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
				return {t, std::move(output_dyn_dims)};
			}
            else if(not s0.dynamic() and not s1.dynamic())
            {
                auto bcast_lens = compute_broadcasted_lens(s0.lens(), s1.lens());
				auto offset = s1.lens().size() - s0.lens().size();
				auto bcast_strides = make_bcast_strides(s1.lens().size(), offset);
				return {t, std::move(bcast_lens), std::move(bcast_strides)};
            }
            else
            {
                MIGRAPHX_THROW(
                    "MULTIBROADCAST: s0 and s1 are not both dynamic or static");
            }
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        return args[0].reshape(dyn_out.computed_shape);
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
