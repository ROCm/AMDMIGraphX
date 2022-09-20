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
#ifndef MIGRAPHX_GUARD_RTGLIB_WHERE_HPP
#define MIGRAPHX_GUARD_RTGLIB_WHERE_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/where.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_where : ternary_device<hip_where, device::where>
{
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4).same_dims();
        auto s1 = inputs.at(1);
        auto s2 = inputs.at(2);
        if(s1 == s2 and s1.packed())
        {
            return s1;
        }
        else if(s1.packed() != s2.packed())
        {
            return s1.packed() ? s1 : s2;
        }
        else if(s1.broadcasted() != s2.broadcasted())
        {
            return s1.broadcasted() ? s2.with_lens(s1.lens()) : s1.with_lens(s1.lens());
        }
        else
        {
            return {s1.type(), s1.lens()};
        }
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
