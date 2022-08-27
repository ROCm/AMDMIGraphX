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
        check_shapes{inputs, *this}.has(1);
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        if(input.lens().empty())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should be > 0");
        }

        if(input.lens().size() > output_lens.size())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should <= output size");
        }

        auto offset = output_lens.size() - input.lens().size();
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] != input.lens()[i] and input.lens()[i] != 1)
            {
                MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(input.lens()) +
                               "} cannot be broadcasted to {" + to_string_range(output_lens) +
                               "}!");
            }
        }

        std::vector<size_t> bcast_strides(output_lens.size(), 0);
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] == input.lens()[i])
            {
                bcast_strides[i + offset] = input.strides()[i];
            }
        }
        return {t, output_lens, bcast_strides};
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
