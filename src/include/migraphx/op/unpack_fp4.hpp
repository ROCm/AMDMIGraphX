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
#ifndef MIGRAPHX_GUARD_OPERATORS_UNPACK_FP4_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNPACK_FP4_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/fp4_casts.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Unpacks fastest dimension of tensor into fp8e4m3fn_type such that the
 * output dimensions are [x_0, ..., 2 * x_pack, ...]
 */
namespace op {
struct unpack_fp4
{
    int64_t axis = -1;

    std::string name() const { return "unpack_fp4"; }

    value attributes() const
    {
        value normalize   = value::object{};
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        check_shapes{inputs, *this}.same_dims().has(1);
        const auto& in_shape = inputs.front();
        if(in_shape.type() != migraphx::shape::fp4x2_type)
        {
            MIGRAPHX_THROW("UNPACK_FP4: Only fp4x2_type is supported for unpacking");
        }
        auto new_lens = in_shape.lens();
        new_lens[axis] *= 2;
        return in_shape.with_lens(migraphx::shape::fp8e4m3fn_type, new_lens);
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        const auto& input = args.front();
        auto in_shape     = input.get_shape();

        migraphx::shape fp8_shape = shape{migraphx::shape::fp8e4m3fn_type, output_shape.lens()};
        argument fp8_arg{fp8_shape};
        auto inp = input.get<uint8_t>();
        fp8_arg.visit([&](auto out) {
            par_for(in_shape.elements(), [&](auto i) {
                auto data_idx = in_shape.multi(i);
                data_idx[axis] *= 2;
                // unpacking 2 unsigned parts
                // unpacking 4 least significant bits first
                uint8_t fp4_val = inp[i];
                out[data_idx]   = fp4_to_fp8(fp4_val);

                data_idx[axis] += 1;
                fp4_val       = fp4_val >> 4u;
                out[data_idx] = fp4_to_fp8(fp4_val);
            });
        });
        return fp8_arg;
    }
};
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
