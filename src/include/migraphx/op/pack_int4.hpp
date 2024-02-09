/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_PACK_INT4_HPP
#define MIGRAPHX_GUARD_OPERATORS_PACK_INT4_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>
#include <migraphx/config.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
struct pack_int4
{
    int64_t axis = -1;

    std::string name() const { return "pack_int4"; }

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
        // remove standard requirement later.
        check_shapes{inputs, *this}.same_dims().has(1).standard();
        auto in_shape = inputs.front();
        if(in_shape.type() != migraphx::shape::uint8_type)
        {
            MIGRAPHX_THROW("PACK_INT4: Only Unsigned Int8 type is supported for packing");
        }
        auto strides  = in_shape.strides();
        auto lens     = in_shape.lens();
        auto new_lens = lens;
        if(lens[axis] % 2 != 0)
        {
            MIGRAPHX_THROW("PACK_INT4: Can not pack axis that has odd lengths");
        }
        new_lens[axis] /= 2;
        return {migraphx::shape::uint8_type, new_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto in_shape = args.front().get_shape();
        args[0].visit([&](auto input) {
            using type = typename decltype(input)::value_type;
            if constexpr(std::is_same<type, uint8_t>{})
            {
                auto output = result.get<uint8_t>();
                par_for(output_shape.elements(), [&](auto i) {
                    auto data_idx          = output_shape.multi(i);
                    auto in_data_multi_idx = data_idx;
                    in_data_multi_idx[axis] *= 2;
                    auto input_val = input[in_data_multi_idx];
                    // mask first 4 bits, keep it little endian.
                    output[i] = 0x0F & input_val;
                    in_data_multi_idx[axis] += 1;
                    input_val = input[in_data_multi_idx];
                    output[i] = (input_val << 4) | (output[i]);
                });
            }
        });
        return result;
    }
};
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
