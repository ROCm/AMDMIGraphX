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
#ifndef MIGRAPHX_GUARD_OPERATORS_GLU_HPP
#define MIGRAPHX_GUARD_OPERATORS_GLU_HPP

#include <migraphx/config.hpp>
#include <migraphx/op/sigmoid.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/slice.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct glu
{
    int64_t dim = -1;
    std::string name() const
    {
        return "glu";
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dim, "dim"));
    }

    auto apply() const
    {
        return [&](auto x) {
            auto lens = x.get_shape().lens();
            int64_t mid = lens[dim] / 2;

            std::vector<int64_t> starts(lens.size(), 0);
            std::vector<int64_t> ends = lens;

            //First Half
            ends[dim] = mid;
            auto first_half = migraphx::op::slice{{dim}, starts, ends}(x);

            //Second Half
            starts[dim] = mid;
            ends[dim] = lens[dim];
            auto second_half = migraphx::op::slice{{dim}, starts, ends}(x);

            auto sigmoid_half = migraphx::op::sigmoid{}(second_half);

            return migraphx::op::mul{}(first_half, sigmoid_half);
        };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
