/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_GPU_STANDARDIZE_OUTPUT_HPP
#define MIGRAPHX_GUARD_GPU_STANDARDIZE_OUTPUT_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct module_pass_manager;

namespace gpu {

/**
 * Pack a returned buffer that is still nonstandard after fusion.
 *
 * This runs on the root module only, after `eliminate_contiguous` and `fuse_ops`
 * and before `replace_allocate`, so it sees the layout that will be aliased onto
 * the output parameter. Control-flow submodules keep their return layouts; their
 * callers already have output buffers for those shapes. A packed, dynamic, or
 * single-element tensor is left alone. A strided tensor gets one `gpu::contiguous`.
 * A tuple return keeps its structure, with each nonstandard tensor leaf packed.
 * A module whose last instruction is the output, with no return instruction, is packed too.
 */
struct MIGRAPHX_GPU_EXPORT standardize_output
{
    std::string name() const { return "gpu::standardize_output"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
