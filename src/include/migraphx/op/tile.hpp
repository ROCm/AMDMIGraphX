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
#ifndef MIGRAPHX_GUARD_OPERATORS_TILE_HPP
#define MIGRAPHX_GUARD_OPERATORS_TILE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * ONNX Tile: repeat the input tensor along each dimension i by repeats[i] times.
 * Repeats are compile-time attributes (from parsed literal repeats tensor).
 */
struct tile
{
    std::vector<int64_t> repeats;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.repeats, "repeats"));
    }

    std::string name() const { return "tile"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& s = inputs.front();
        if(repeats.size() != s.ndim())
        {
            MIGRAPHX_THROW(
                "TILE: repeats length must match input rank (repeats size " +
                std::to_string(repeats.size()) + " vs ndim " + std::to_string(s.ndim()) + ")");
        }
        for(int64_t r : repeats)
        {
            if(r <= 0)
                MIGRAPHX_THROW("TILE: each repeat count must be positive");
        }

        if(s.dynamic())
        {
            auto dds = s.dyn_dims();
            for(std::size_t i = 0; i < dds.size(); ++i)
            {
                const auto rv = static_cast<std::size_t>(repeats[i]);
                dds[i]        = dds[i] * shape::dynamic_dimension{rv, rv};
            }
            return {s.type(), std::move(dds)};
        }

        const auto& lens = s.lens();
        std::vector<std::size_t> out_lens(lens.size());
        for(std::size_t i = 0; i < lens.size(); ++i)
        {
            out_lens[i] = lens[i] * static_cast<std::size_t>(repeats[i]);
        }
        return {s.type(), std::move(out_lens)};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        const auto& in_shape = args[0].get_shape();
        const auto& in_lens  = in_shape.lens();

        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(output.get_shape(), [&](const auto& oidx) {
                std::vector<std::size_t> iidx(in_lens.size());
                for(std::size_t d = 0; d < in_lens.size(); ++d)
                {
                    iidx[d] = oidx[d] % in_lens[d];
                }
                output(oidx.begin(), oidx.end()) = input(iidx.begin(), iidx.end());
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
