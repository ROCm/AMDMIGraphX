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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SYSTEM_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SYSTEM_HPP

#include <migraphx/config.hpp>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// The physical cores this process may run on, counting the hardware threads of a core once. Where
/// the core topology is unknown, which is everywhere but Linux, each logical CPU counts as a core.
MIGRAPHX_EXPORT std::size_t physical_cpu_cores();

/// Hands the memory the allocator keeps after it is freed back to the operating system. Does
/// nothing where the C library can't.
MIGRAPHX_EXPORT void trim_heap();

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
