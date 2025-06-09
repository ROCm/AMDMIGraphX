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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

namespace {
// roundTiesToEven
constexpr uint8_t cast_to_fp4(float f_x)

    uint32_t x = migraphx::bit_cast<uint32_t>(f_x);

} // namespace

struct pact_fp4
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
        return {migraphx::shape::mxfp4_type, new_lens};
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        auto input    = args.front();
        auto in_shape = input.get_shape();

        auto uint8_shape = shape{migraphx::shape::uint8_type, output_shape.lens()};
        argument result{uint8_shape};
        visit(result)([&](auto res) {
            visit(input)([&](auto inp) {
                par_for(output_shape.elements(), [&](auto i) {
                    uint8_t lowest_fp4 = 0xf;
                    uint8_t max_fp4    = 0x7;
                });
            });
        });
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
