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
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_ENTRY_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_ENTRY_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/compiled_code.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/value.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/// What gets stored for one compiled kernel. The op name, problem and solution are stored for
/// offline inspection; only the key is checked when an entry is loaded.
///
/// This lives in its own header, rather than nested inside binary_cache, so that the
/// binary_cache_backend interface can name it without including binary_cache.hpp -- which in
/// turn includes the backend header. Same reason cache_device_key.hpp exists for the problem
/// cache. binary_cache::entry remains an alias for it.
struct binary_cache_entry
{
    std::string key     = {};
    std::string op_name = {};
    value problem       = {};
    value solution      = {};
    compiled_code code  = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.key, "key"),
                    f(self.op_name, "op_name"),
                    f(self.problem, "problem"),
                    f(self.solution, "solution"),
                    f(self.code, "code"));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_ENTRY_HPP
