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

#ifndef MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct MIGRAPHX_EXPORT dominator_info
{
    bool strictly_dominate(instruction_ref ins1, instruction_ref ins2) const;

    std::unordered_map<instruction_ref, instruction_ref> ins2idom;
};

struct MIGRAPHX_EXPORT dominator_options
{
    bool ignore_constants = false;
};

MIGRAPHX_EXPORT dominator_info compute_dominator(const module& m, dominator_options options = {});
// MIGRAPHX_EXPORT dominator_info compute_dominator_naive(const module& m);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
