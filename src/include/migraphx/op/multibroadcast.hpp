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
#include <migraphx/dyn_output.hpp>
#include <migraphx/common.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Broadcast multiple dimensions between two tensors.
 * Two versions of this operator: one input and two inputs.
 * One input version uses output_lens attribute and broadcasts to it.
 * Two inputs version broadcasts both inputs to the common shape at evaluation time.
 */
struct multibroadcast
{
    std::vector<std::size_t> output_lens = {};

    // optional attribute
    std::vector<shape::dynamic_dimension> output_dyn_dims = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "out_lens"), f(self.output_dyn_dims, "out_dyn_dims"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);

        auto t  = inputs.at(0).type();
        auto s0 = inputs.at(0);

        if(s0.max_lens().empty())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should be > 0");
        }

        auto make_bcast_strides = [&](std::vector<std::size_t> bcast_lens, std::size_t offset) {
            std::vector<size_t> bcast_strides(bcast_lens.size(), 0);
            for(std::ptrdiff_t i = s0.lens().size() - 1; i >= 0; i--)
            {
                if(bcast_lens[i + offset] == s0.lens()[i])
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
                MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should <= output size");
            }

            auto offset = output_lens.size() - s0.lens().size();
            for(std::ptrdiff_t i = s0.lens().size() - 1; i >= 0; i--)
            {
                if(output_lens[i + offset] != s0.lens()[i] and s0.lens()[i] != 1)
                {
                    MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(s0.lens()) +
                                   "} cannot be broadcasted to {" + to_string_range(output_lens) +
                                   "}!");
                }
            }

            auto bcast_strides = make_bcast_strides(output_lens, offset);
            return {t, output_lens, std::move(bcast_strides)};
        }
        else
        {
            // two inputs
            auto s1 = inputs.at(1);
            if(s0.dynamic() or s1.dynamic())
            {
                if(not output_dyn_dims.empty())
                {
                    return {t, output_dyn_dims};
                }
                return {t, compute_broadcasted_dyn_dims(s0, s1)};
            }
            else
            {
                auto bcast_lens    = compute_broadcasted_lens(s0.lens(), s1.lens());
                auto offset        = bcast_lens.size() - s0.lens().size();
                auto bcast_strides = make_bcast_strides(bcast_lens, offset);
                return {t, std::move(bcast_lens), std::move(bcast_strides)};
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
