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

#ifndef MIGRAPHX_GUARD_OPERATORS_RANDOM_SEED_HPP
#define MIGRAPHX_GUARD_OPERATORS_RANDOM_SEED_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/reflect.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 *    Generates a random seed for the use of random number generators.  Generating the seed
 * at runtime guarantees there will be a different random sequence on every execution.
 * This operation has no inputs or attributes, and outputs an unsigned integer tensor with
 * a single value.
 */
struct random_seed
{
    shape::type_t dtype = shape::type_t::uint32_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dtype, "dtype"));
    }

    std::string name() const { return "random_seed"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        (void)inputs;
        return migraphx::shape(dtype, {1});
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        (void)args;
        argument result(output_shape);

        result.visit([&](auto output) {
            std::generate(output.begin(), output.end(), [&]() {
                return uint32_t(std::chrono::system_clock::now().time_since_epoch().count());
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
