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
#ifndef MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP
#define MIGRAPHX_GUARD_RTGLIB_ALGORITHM_HPP

#include <algorithm>
#include <numeric>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator, class Output, class Predicate, class F>
void transform_if(Iterator start, Iterator last, Output out, Predicate pred, F f)
{
    while(start != last)
    {
        if(pred(*start))
        {
            *out = f(*start);
            ++out;
        }
        ++start;
    }
}

template <class Iterator, class T, class BinaryOp, class UnaryOp>
T transform_accumulate(Iterator first, Iterator last, T init, BinaryOp binop, UnaryOp unaryop)
{
    return std::inner_product(
        first, last, first, init, binop, [&](auto&& x, auto&&) { return unaryop(x); });
}

template <class Iterator, class Output, class Predicate>
void group_by(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::partition(start, last, [&](auto&& x) { return pred(x, *start); });
        out(start, it);
        start = it;
    }
}

template <class Iterator, class Output, class Predicate>
void group_unique(Iterator start, Iterator last, Output out, Predicate pred)
{
    while(start != last)
    {
        auto it = std::find_if(start, last, [&](auto&& x) { return not pred(*start, x); });
        out(start, it);
        start = it;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
