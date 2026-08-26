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
#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/compile_src.hpp>
#include <migraphx/env.hpp>
#include <migraphx/functional.hpp>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef MIGRAPHX_USE_HIPRTC
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_HIPRTC);
#endif

struct hiprtc_src_file
{
    hiprtc_src_file() = default;
    hiprtc_src_file(const src_file& s) : path(s.path.string()), content(s.content) {}
    std::string path;
    std::string content;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.path, "path"), f(self.content, "content"));
    }
};

/// What the compiler that turns kernel source into code objects reports about itself.
struct hip_compiler_info
{
    /// __clang_major__ and __clang_minor__, kept separate so the cache directory stays readable.
    std::string major = {};
    std::string minor = {};
    /// The full __clang_version__ string, which also names the source revision.
    std::string version = {};

    std::vector<std::string> flags = {};

    bool empty() const { return version.empty(); }
};

std::vector<std::string> compile_hip_options(const std::vector<std::string>& params,
                                             const std::string& arch);

/**
 * Ask the device compiler what version it is.
 *
 * It need not be the compiler this library was built with, so the answer comes from compiling a
 * probe and reading back what it recorded. The result is determined once and reused; it is
 * empty if the compiler could not be asked.
 */
MIGRAPHX_GPU_EXPORT const hip_compiler_info& hip_compiler_version();

MIGRAPHX_GPU_EXPORT bool hip_can_compile(const std::string& src,
                                         const std::vector<std::string>& flags);

MIGRAPHX_GPU_EXPORT bool hip_has_flags(const std::vector<std::string>& flags);

MIGRAPHX_GPU_EXPORT std::vector<std::vector<char>>
compile_hip_src_with_hiprtc(std::vector<hiprtc_src_file> srcs,
                            const std::vector<std::string>& params,
                            const std::string& arch,
                            bool quiet = false);

MIGRAPHX_GPU_EXPORT std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs,
                const std::vector<std::string>& params,
                const std::string& arch,
                bool quiet = false);

MIGRAPHX_GPU_EXPORT std::string enum_params(std::size_t count, std::string param);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
