/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * Packs fastest dimension of tensor into fp4x2_type such that the
 * output dimensions are [x_0, ..., x_pack/2, ...]
 */
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
        auto new_lens = in_shape.lens();
        if(new_lens.at(axis) % 2 != 0)
        {
            std::stringstream msg;
            msg << "PACK_FP4: Can not pack along axis of odd length (" << new_lens.at(axis) << ")";
            MIGRAPHX_THROW(msg.str());
        }
        new_lens[axis] /= 2;
        return in_shape.with_lens(migraphx::shape::fp4x2_type, new_lens);
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        const auto& input = args.front();
        auto in_shape = input.get_shape();
        argument result{output_shape};
        auto out = result.get<uint8_t>();
        input.visit([&](auto inp) {
            par_for(output_shape.elements(), [&](auto i) {
                using inp_type         = typename decltype(inp)::value_type;
                auto data_idx          = output_shape.multi(i);
                auto in_data_multi_idx = data_idx;
                in_data_multi_idx[axis] *= 2;
                inp_type inp_val0 = inp[in_data_multi_idx];
                in_data_multi_idx[axis] += 1;
                inp_type inp_val1 = inp[in_data_multi_idx];
                uint8_t out_val0  = cast_to_fp4(inp_val0);
                uint8_t out_val1  = cast_to_fp4(inp_val1);
                // NOTE: integral promotion occurs when bitshifting for uint8_t
                out[i] =
                    static_cast<uint8_t>(out_val1 << 4u) | static_cast<uint8_t>(out_val0 & 0xFu);
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
