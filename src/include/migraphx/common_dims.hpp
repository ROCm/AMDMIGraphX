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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP

#include <migraphx/config.hpp>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// This will compute a higher dimensional space that will preserve the axes
/// for both sets of dimensions. Two axes_maps are provided for each of the
/// dims that will map the axis to the axes that are used by the result of
/// common_dims.
struct MIGRAPHX_EXPORT common_dims
{
    static common_dims compute(const std::vector<std::size_t>& dims1,
                               const std::vector<std::size_t>& dims2);
    std::vector<std::size_t> dims;
    std::vector<std::vector<std::size_t>> axes_map1;
    std::vector<std::vector<std::size_t>> axes_map2;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMMON_DIMS_HPP
