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

/**
 * Random Uniform distribution operator.  Given a shape, populate it with random values.
 *
 *      Inputs:   any tensor shape.
 *      Attributes:  TBD
 *
        Output:   Same shape.
 *
*/
#ifndef MIGRAPHX_GUARD_OPERATORS_MULTINOMIAL_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTINOMIAL_HPP

#include <migraphx/check_shapes.hpp>
// #include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
// #include <migraphx/reflect.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rand_uniform
{
    uint32_t sample_size = {20};
    uint32_t seed = {0};
    shape::type_t dtype = shape::type_t::float_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dtype, "dtype"), f(self.sample_size, "sample_size"), f(self.seed, "seed"));
    }

    value attributes() const
    {
        return {{"sample_size", sample_size}, {"seed", seed}};
    }


    std::string name() const { return "rand_uniform"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto s = inputs.front();
        if(s.dynamic())
        {
            return s;
        }
        else if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }


    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        size_t index(dyn_out.computed_shape.elements());
        // Use of our visitor and par_for replaces a call like 
        //   std::vector<float> rand_samples(sample_size);
        //   std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });

        result.visit([&](auto output) {
            par_for(sample_size, [&](auto i)
            {
                    output[i] = dis(gen);
                    // output[i] = rand_samples[i];
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
