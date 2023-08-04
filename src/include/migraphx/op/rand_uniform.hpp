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
 *      Inputs:   (1) the shape of the set to be populated.
 *                (2) randomization seed.  Optional--if not given, a seed will be generated
 *                    automatically, for nonrepeatable random results.
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
    // uint32_t sample_size = {20};
    uint32_t seed      = {0};
    bool use_auto_seed = false;
    float range_min    = 0.0f;
    float range_max    = 1.0f;

    // From Onnx RandomUniform:
    // dtype : int (default is 1) The data type for the elements of the output tensor. currently
    // float only. high : float (default is 1.0) Upper boundary of the output values. low : float
    // (default is 0.0) Lower boundary of the output values. seed : float (Optional) Seed to the
    // random generator, if not specified we will auto generate one. shape : list of ints (required)
    // The shape of the output tensor.

    // In Onnx, the size of array to fill is given by

    // TODO:  consider removing this and simply using the type of the passed argument.
    //  The only bar to doing this currently is that we can't create random integers within the
    // current bounds of (0, 1).
    shape::type_t dtype             = shape::type_t::float_type;
    std::vector<size_t> output_lens = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dtype, "dtype"),
                    f(self.output_lens, "output_lens"),
                    f(self.seed, "seed"),
                    f(self.use_auto_seed, "use_auto_seed"));
    }

    // value attributes() const { return {{"sample_size", sample_size}, {"seed", seed}}; }

    std::string name() const { return "rand_uniform"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);

        if(inputs.size() > 1 and inputs.at(1).element_space() > 0 and
           inputs.at(1).type() != shape::type_t::uint32_type)
            MIGRAPHX_THROW("RAND_UNIFORM:  Input 2 (seed) must have type unsigned int");
        auto s = inputs.front();
        if(s.dynamic())
        {
            return s.with_type(dtype);
        }
        else
        {
            return s.with_lens(s.lens()).with_type(dtype);
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        (void)args; // suppress compiler warning
        argument result{dyn_out.computed_shape};

        auto local_seed(seed);
        if(use_auto_seed)
            local_seed = std::chrono::system_clock::now().time_since_epoch().count();
        else
        {
            if(args.size() > 1)
            {
                if(args.at(1).get_shape().element_space() > 0)
                {
                    visit_all(args[1])([&](auto data) { local_seed = data[0]; });
                }
                else // This is a bit of an Easter Egg.
                     // If a seed argument was given but it has a 0-size shape at
                     // inference time, also obtain a seed from the system clock:
                    local_seed = std::chrono::system_clock::now().time_since_epoch().count();
            }
        }
        // If a seed argument was not defined, use the value from the seed attribute,
        // or the default.

        std::mt19937 gen(local_seed);
        std::uniform_real_distribution<> dis(range_min, range_max);
        result.visit([&](auto output) {
            std::generate(output.begin(), output.end(), [&]() { return dis(gen); });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
