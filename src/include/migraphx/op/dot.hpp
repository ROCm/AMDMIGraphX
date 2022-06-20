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
#ifndef MIGRAPHX_GUARD_OPERATORS_DOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_DOT_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gemm.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct dot
{
    std::string name() const { return "dot"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.same_type().has(2);
        const shape& a = inputs.at(0);
        const shape& b = inputs.at(1);
        auto t         = a.type();

        if(!std::all_of(inputs.begin(), inputs.end(), [](auto s) { return s.lens().size() >= 2; }))
        {
            MIGRAPHX_THROW("DOT: dot only accept 2 or more dims operands");
        }

        // only handle the case that the batch size of a and b are the same
        if(!std::equal(
               a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2, b.lens().rend()))
        {
            MIGRAPHX_THROW("DOT: batch size of A and B mismatch: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
        {
            MIGRAPHX_THROW("DOT: inner dimensions do not match: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        }

        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        return {t, out_lens};
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result = argument{output_shape};
        visit_all(result, args[0], args[1])(
            [&](auto cmat, auto amat, auto bmat) { gemm(cmat, amat, bmat, 1.0f, 0.0f); });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
