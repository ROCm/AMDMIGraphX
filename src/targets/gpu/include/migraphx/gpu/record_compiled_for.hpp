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
#ifndef MIGRAPHX_GUARD_GPU_RECORD_COMPILED_FOR_HPP
#define MIGRAPHX_GUARD_GPU_RECORD_COMPILED_FOR_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/device_description.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/value.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct module_pass_manager;
struct compile_options;

namespace gpu {

struct context;

/// Record the device the program is compiled for as a comment tagged "gpu_target", and the options
/// it is compiled with as a comment tagged "gpu_compile_options", so it can be recompiled. The
/// options are only for reference: recompile does not read them and leaves them as recorded. Runs
/// last, since passes such as schedule do not expect a comment in the graph.
struct MIGRAPHX_GPU_EXPORT record_compiled_for
{
    context* ctx = nullptr;
    /// The options as made by compile_options_value, or null to record only the device
    value options = {};
    std::string name() const { return "gpu::record_compiled_for"; }
    void apply(module_pass_manager& mpm) const;
};

/// The compile options and the GPU backend options as a value, to record with the program
MIGRAPHX_GPU_EXPORT value compile_options_value(const compile_options& options,
                                                const value& backend_options);

/// The device recorded by record_compiled_for, or nothing
MIGRAPHX_GPU_EXPORT optional<device_description> get_compiled_for(const module& m);

/// The options recorded by record_compiled_for, or null
MIGRAPHX_GPU_EXPORT value get_compile_options(const module& m);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_RECORD_COMPILED_FOR_HPP
