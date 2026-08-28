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
#ifndef MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP
#define MIGRAPHX_GUARD_OPERATORS_NONZERO_HPP

#include <migraphx/shape_for_each.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <cmath>
#include <cstdint>
#include <utility>

/**
 *  nonzero(data);
 *  Outputs tuple of {indices, num_nonzero}.
 *  `indices` are padded out to the most elements the input shape allows.
 *  `num_nonzero` tells how many of the columns hold a real index.
 */
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct nonzero
{
    std::string name() const { return "nonzero"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Pad the indices for the largest input the shape allows, so a dynamic input still gets
        // a fixed output buffer. num_nonzero says how many of the columns are real.
        shape max_input{inputs[0].type(), inputs[0].max_lens()};
        shape s_ind{shape::int64_type, {inputs[0].ndim(), max_input.elements()}};
        shape s_num_nonzero{shape::int64_type, {1}};
        return shape({s_ind, s_num_nonzero});
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto s             = args.front().get_shape();
        const auto& vec_ss = output_shape.sub_shapes();
        argument result{vec_ss.front()};
        argument num_nonzero_result{vec_ss.back()};
        auto output = result.get<std::int64_t>();
        std::fill(output.begin(), output.end(), 0);
        std::size_t nonzero_idx = 0;
        args.front().visit([&](auto v) {
            shape_for_each(s, [&](const auto& idx_v) {
                if(not float_equal(v[idx_v], 0))
                {
                    auto out_idx = nonzero_idx++;
                    par_for(idx_v.size(), [&](auto i) { output(i, out_idx) = idx_v[i]; });
                }
            });
        });
        num_nonzero_result.visit([&](auto num_nonzero) { num_nonzero[0] = nonzero_idx; });

        return {{result, num_nonzero_result}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
