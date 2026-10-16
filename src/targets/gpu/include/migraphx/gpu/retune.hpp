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
#ifndef MIGRAPHX_GUARD_GPU_RETUNE_HPP
#define MIGRAPHX_GUARD_GPU_RETUNE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/value.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

namespace gpu {

struct context;

/**
 * Recompile the gpu::code_object kernels of a compiled program from their recipes for the
 * device of ctx, in place. The graph, literals, memory plan and schedule are not changed.
 *
 * options holds "from", the serialized context of the program being retuned, and optionally
 * "force" to recompile kernels that are already at the requested tuning level.
 */
struct MIGRAPHX_GPU_EXPORT retune_ops
{
    context* ctx         = nullptr;
    value options        = value::object{};
    bool exhaustive_tune = false;
    std::string name() const { return "gpu::retune_ops"; }
    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_RETUNE_HPP
