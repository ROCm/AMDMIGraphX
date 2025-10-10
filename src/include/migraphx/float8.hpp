/* ************************************************************************
 * Copyright (C) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
#define MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP

// We are clipping/saturation in down conversion by default. Unclipped version is not tested and
// shouldn't be used without having enough tests.
// logic is based on clipping table from here : https://onnx.ai/onnx/technical/float8.html#cast
// NOLINTNEXTLINE
#define MIGRAPHX_F8_DOWNCAST_CLIPPING 1

#include <migraphx/config.hpp>
#include <migraphx/generic_float.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using fp8e4m3fn   = migraphx::generic_float<3, 4, 1>;
using fp8e5m2     = migraphx::generic_float<2, 5, 1>;
using fp8e4m3fnuz = migraphx::generic_float<3, 4, 2>;
using fp8e5m2fnuz = migraphx::generic_float<2, 5, 2>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_RTGLIB_FLOAT8_HPP
