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
 *
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_GATHER_DOT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_GATHER_DOT_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/// Rewrite dots whose matrices are gathered per row from a table, eg the
/// weights of the experts selected for each token of a mixture of experts
/// layer, into one batched dot over the whole table followed by a gather of
/// the selected rows of the result. With enough rows per table entry the
/// gathers read every matrix several times, whereas the batched dot reads
/// each matrix once.
struct MIGRAPHX_EXPORT rewrite_gather_dot
{
    /// The number of gathered rows per table entry from which the batched dot
    /// is used. This is the crossover where the gathered dots become slower
    /// than a dot over the whole table, which depends on the target.
    float min_rows_per_entry = 1.0f;

    std::string name() const { return "rewrite_gather_dot"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_GATHER_DOT_HPP
