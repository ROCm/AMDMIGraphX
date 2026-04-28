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
#ifndef MIGRAPHX_GUARD_GPU_PIPELINE_FACTORY_HPP
#define MIGRAPHX_GUARD_GPU_PIPELINE_FACTORY_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/pass.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
struct MIGRAPHX_GPU_EXPORT pipeline_factory
{
    migraphx::context* gctx_ptr = nullptr;
    compile_options options     = {};

    migraphx::context* get_generic_context() const { return gctx_ptr; }

    context* get_context() const;

    std::vector<pass> dynamic_shapes_pipeline() const;
    std::vector<pass> required_pipeline() const;
    std::vector<pass> optimize_rewrite_pipeline() const;
    std::vector<pass> prefuse_pipeline() const;
    std::vector<pass> rewrite_simplify_pipeline() const;
    std::vector<pass> fusion_pipeline() const;
    std::vector<pass> backend_pipeline() const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PIPELINE_FACTORY_HPP
