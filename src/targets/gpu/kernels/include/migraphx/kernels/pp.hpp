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
#ifndef MIGRAPHX_GUARD_KERNELS_PP_HPP
#define MIGRAPHX_GUARD_KERNELS_PP_HPP

#define MIGRAPHX_PP_PRIMITIVE_CAT(x, y) x##y
#define MIGRAPHX_PP_CAT(x, y) MIGRAPHX_PP_PRIMITIVE_CAT(x, y)

#define MIGRAPHX_PP_EAT(...)
#define MIGRAPHX_PP_EXPAND(...) __VA_ARGS__

#define MIGRAPHX_PP_REPEAT0(m, ...) m(0, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT1(m, ...) MIGRAPHX_PP_REPEAT0(m, __VA_ARGS__) m(1, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT2(m, ...) MIGRAPHX_PP_REPEAT1(m, __VA_ARGS__) m(2, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT3(m, ...) MIGRAPHX_PP_REPEAT2(m, __VA_ARGS__) m(3, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT4(m, ...) MIGRAPHX_PP_REPEAT3(m, __VA_ARGS__) m(4, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT5(m, ...) MIGRAPHX_PP_REPEAT4(m, __VA_ARGS__) m(5, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT6(m, ...) MIGRAPHX_PP_REPEAT5(m, __VA_ARGS__) m(6, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT7(m, ...) MIGRAPHX_PP_REPEAT6(m, __VA_ARGS__) m(7, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT8(m, ...) MIGRAPHX_PP_REPEAT7(m, __VA_ARGS__) m(8, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT9(m, ...) MIGRAPHX_PP_REPEAT8(m, __VA_ARGS__) m(9, __VA_ARGS__)
#define MIGRAPHX_PP_REPEAT10(m, ...) MIGRAPHX_PP_REPEAT9(m, __VA_ARGS__) m(10, __VA_ARGS__)

#define MIGRAPHX_PP_REPEAT(n, m, ...) \
    MIGRAPHX_PP_PRIMITIVE_CAT(MIGRAPHX_PP_REPEAT, n)(m, __VA_ARGS__)

namespace migraphx {

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PP_HPP
