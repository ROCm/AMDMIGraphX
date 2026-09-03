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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_LAUNCH_DIMS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_LAUNCH_DIMS_HPP

#include <migraphx/config.hpp>
#include <cstddef>
#include <array>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct launch_dims
{
    launch_dims(std::size_t x) : dims{x, 1, 1} {}
    launch_dims(std::size_t x, std::size_t y) : dims{x, y, 1} {}
    launch_dims(std::size_t x, std::size_t y, std::size_t z) : dims{x, y, z} {}

    std::size_t x() const { return dims[0]; }
    std::size_t y() const { return dims[1]; }
    std::size_t z() const { return dims[2]; }

    std::array<std::size_t, 3> dims;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
