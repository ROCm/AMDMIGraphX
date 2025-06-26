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
#ifndef MIGRAPHX_GUARD_OPERATORS_PACK_FP4_HPP
#define MIGRAPHX_GUARD_OPERATORS_PACK_FP4_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/fp4_casts.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pack_fp4
{
    int64_t axis = -1;

    std::string name() const { return "pack_fp4"; }

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
        auto new_lens        = in_shape.lens();
        if(new_lens[axis] % 2 != 0)
        {
            MIGRAPHX_THROW("PACK_FP4: Can not pack axis that has odd lengths");
        }
        new_lens[axis] /= 2;
        return {migraphx::shape::packed_fp4_type, new_lens};
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        auto input    = args.front();
        auto in_shape = input.get_shape();

        migraphx::shape uint8_shape = shape{migraphx::shape::uint8_type, output_shape.lens()};
        argument uint8_arg{uint8_shape};
        uint8_arg.visit([&](auto out) {
            input.visit([&](auto inp) {
                par_for(output_shape.elements(), [&](auto i) {
                    using inp_type         = typename decltype(inp)::value_type;
                    auto data_idx          = output_shape.multi(i);
                    auto in_data_multi_idx = data_idx;
                    in_data_multi_idx[axis] *= 2;
                    inp_type inp_val0 = inp[in_data_multi_idx];
                    in_data_multi_idx[axis] += 1;
                    inp_type inp_val1 = inp[in_data_multi_idx];
                    uint8_t out_val0  = float_to_fp4(inp_val0);
                    uint8_t out_val1  = float_to_fp4(inp_val1);
                    out[i]            = (out_val0 << 4) | (out_val1 & 0xf);
                });
            });
        });
        migraphx::argument result =
            uint8_arg.reshape({migraphx::shape::packed_fp4_type, output_shape.lens()});
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
