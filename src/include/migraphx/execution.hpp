/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_EXECUTION_HPP
#define MIGRAPHX_GUARD_EXECUTION_HPP

#ifdef DEBIAN_DISTRO
#include <execution>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class ForwardIt1, class ForwardIt2, class UnaryPredicate> // NOLINT
ForwardIt2 copy_if(ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first, UnaryPredicate pred)
{
#ifdef DEBIAN_DISTRO
    return std::copy_if(std::execution::par, first, last, d_first, pred);
#else
    return std::copy_if(first, last, d_first, pred);
#endif
}

template <class RandomIt, class Compare>
constexpr void sort(RandomIt first, RandomIt last, Compare comp)
{
#ifdef DEBIAN_DISTRO
    return std::sort(std::execution::par, first, last, comp);
#else
    return std::sort(first, last, comp);
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
