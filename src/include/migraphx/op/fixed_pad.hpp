/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_FIXED_PAD_HPP
#define MIGRAPHX_GUARD_OPERATORS_FIXED_PAD_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Pads the given input to the dimensions given in the `output_lens` attribute.
 * The main use for this op versus the standard pad op is that it can
 * accept a dynamic input shape and convert it to a padded static shape.
 */
struct fixed_pad
{
    std::vector<size_t> output_lens = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "output_lens"));
    }

    std::string name() const { return "fixed_pad"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& s0 = inputs.front();
        if(s0.ndim() != output_lens.size())
        {
            MIGRAPHX_THROW("FIXED_PAD: input number of dimensions should match output_lens size");
        }
        if(s0.dynamic())
        {
            for(auto i = 0; i < s0.ndim(); i++)
            {
                if(output_lens[i] < s0.dyn_dims()[i].min)
                    MIGRAPHX_THROW("FIXED_PAD: padding to size smaller than min dyn dim");
                if(output_lens[i] > s0.dyn_dims()[i].max)
                    MIGRAPHX_THROW("FIXED_PAD: padding to size larger than max dyn dim");
            }
        }
        else
        {
            if(std::mismatch(s0.lens().begin(),
                             s0.lens().end(),
                             output_lens.begin(),
                             [&](auto in_dim, auto out_dim) { return in_dim <= out_dim; })
                   .first != s0.lens().end())
            {
                MIGRAPHX_THROW("FIXED_PAD: output lens are smaller than input lens");
            }
        }

        return {s0.type(), output_lens};
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        const auto& input_arg = args.front();
        auto input_shape      = input_arg.get_shape();
        if(input_shape == output_shape)
            return input_arg;

        if(std::mismatch(input_shape.lens().begin(),
                         input_shape.lens().end(),
                         output_shape.lens().begin(),
                         [&](auto in_dim, auto out_dim) { return in_dim <= out_dim; })
               .first != input_shape.lens().end())
        {
            MIGRAPHX_THROW("COMPUTE_FIXED_PAD: output lens are smaller than input lens");
        }

        argument out{output_shape};
        visit_all(out, input_arg)([&](auto output, auto input) {
            par_for(input_shape.elements(), [&](auto i) {
                auto idx    = input_shape.multi(i);
                output[idx] = input[idx];
            });
        });

        return out;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
