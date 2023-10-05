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
#ifndef MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP
#define MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Static allocate:
 * No inputs: allocate()
 * `s` attribute set to the static output shape of the buffer.
 *
 * Dynamic allocate:
 * One input: allocate(output_dims)
 * Output_dims are the output buffer dimensions and has a static shape.
 * Either `s` or `buf_type` must be set to calculate the dynamic output shape at compute time.
 * If `buf_type` is set, the compute_shape() of allocate at compile time will have
 * dynamic_dimensions from {0, max_int} with rank = output_dims.ndim(). If `s` is set then the
 * compute_shape() will output `s`; `s` should be a dynamic shape.
 */
struct allocate
{
    shape s{};
    // for dynamic allocate to set the buffer type
    shape::type_t buf_type = shape::half_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.buf_type, "buf_type"));
    }

    std::string name() const { return "allocate"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(s != shape())
        {
            if(inputs.size() == 1)
            {
                migraphx::check_shapes{inputs, *this, false}.only_dims(1);
            }
            else
            {
                migraphx::check_shapes{inputs, *this, false}.has(0);
            }
            return s;
        }
        else
        {
            migraphx::check_shapes{inputs, *this, false}.has(1).only_dims(1);
            const auto& out_dims = inputs.at(0);
            std::size_t max_val = std::numeric_limits<std::size_t>::max();
            std::vector<shape::dynamic_dimension> dyn_dims(out_dims.lens().at(0),
                                                           shape::dynamic_dimension{0, max_val});
            return {buf_type, dyn_dims};
        }
    }
    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        if(args.empty())
        {
            return {output_shape};
        }
        else
        {
            std::vector<std::size_t> output_dims(output_shape.ndim());
            args.at(0).visit([&](auto a) { output_dims.assign(a.begin(), a.end()); });
            return {shape{buf_type, output_dims}};
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
