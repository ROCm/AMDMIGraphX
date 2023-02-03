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
#ifndef MIGRAPHX_GUARD_OPERATORS_OP_NORMALIZE_ATTRIBUTE_HPP
#define MIGRAPHX_GUARD_OPERATORS_OP_NORMALIZE_ATTRIBUTE_HPP

#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * normalize_attribute settings:
 * 1. use_input (default) vs. use_output: changes the rank of the attribute. use_input -> lens.size(), use_output -> lens.size() + vec.size(). Used in unsqueeze.
 * 2. use_rank (default) vs use_len: use_rank sets max value of attribute as the rank of lens. use_lens sets the max value as the corresponding values in lens at as axes
 * use_len is for something like slicing dimensions
 * use_rank is for max axes
 * 3. clip_min vs. not_clip_min (default): clip values less than the minimum to the minimum or not
 * 4. include_min vs. exclude_min (default): include the minimum value or exclude it for range checking and clipping
 * 5. clip_max vs not_clip_max (default): clip values greater than the maximum or not
 * 6. include_max vs. exclude_max (default): include or exclude the maximum value for range checking and clipping
 * 7. normalize_padding: to normalize the padding to 2*(pad ndim) or not
 */
enum class normalize_attribute
{
    use_output,
    use_len,
    clip_max,
    clip_min,
    include_max,
    include_min,
    normalize_padding
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
