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
#ifndef MIGRAPHX_GUARD_RTGLIB_FREEZE_DYN_DIM_HPP
#define MIGRAPHX_GUARD_RTGLIB_FREEZE_DYN_DIM_HPP

#include <string>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Freeze a single dynamic dimension to a static size at compile time.
 *
 * When MIGRAPHX_DYN_DIM_FREEZE_TO=<N> is set (non-zero), every parameter
 * with a non-fixed dynamic dimension is rewritten in place to a static
 * shape at size N.  The rest of the compilation pipeline (split_single_
 * dyn_dim and the GPU lowering) then sees a fully-static program -- no
 * submodules, no select_module, no per-inference dispatch.
 *
 * This gives true static-shape inference latency at the cost of being
 * committed to a single representative input size per compiled engine.
 * For multi-bucket workloads, compile multiple engines (one per bucket)
 * and dispatch caller-side; see docs for the recommended pattern.
 */
struct MIGRAPHX_EXPORT freeze_dyn_dim
{
    std::string name() const { return "freeze_dyn_dim"; }
    void apply(module_pass_manager&) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
