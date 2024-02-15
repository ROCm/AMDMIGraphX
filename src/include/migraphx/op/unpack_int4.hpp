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
#ifndef MIGRAPHX_GUARD_OPERATORS_UNPACK_INT4_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNPACK_INT4_HPP

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
struct unpack_int4
{
    int64_t axis = -1;

    std::string name() const { return "unpack_int4"; }

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
        auto in_shape = inputs.front();
        if(in_shape.type() != migraphx::shape::uint8_type)
        {
            MIGRAPHX_THROW("UNPACK_INT4: Only Unsigned Int8 type is supported for unpacking");
        }
        auto new_lens = in_shape.lens();
        new_lens[axis] *= 2;
        return {migraphx::shape::uint8_type, new_lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto in_shape = args.front().get_shape();
        result.visit([&](auto output) {
            using type = typename decltype(output)::value_type;
            if constexpr(std::is_same<type, uint8_t>{})
            {
                auto input = args[0].get<uint8_t>();
                par_for(in_shape.elements(), [&](auto i) {
                    auto data_idx           = in_shape.multi(i);
                    auto out_data_multi_idx = data_idx;
                    out_data_multi_idx[axis] *= 2;
                    auto input_val = input[data_idx];
                    // mask first 4 bits, packing is assumed to be little endian
                    output[out_data_multi_idx] = static_cast<uint8_t>(0x0F) & input_val;
                    out_data_multi_idx[axis] += 1;
                    output[out_data_multi_idx] = input_val >> 4; // NOLINT
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
