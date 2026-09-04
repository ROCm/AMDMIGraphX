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
#ifndef MIGRAPHX_GUARD_RTGLIB_SPLIT_SYM_DIM_HPP
#define MIGRAPHX_GUARD_RTGLIB_SPLIT_SYM_DIM_HPP

#include <cstddef>
#include <string>
#include <migraphx/pass_manager.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Precompile a graph carrying symbolic (dynamic) dimensions into a small set of
 * fully-static submodules, one per optimal size combination, referenced by a
 * `select_module`. Each submodule pads its symbolic inputs up to a fixed
 * optimal size and runs the ahead-of-time-compiled static body. Runtime
 * symbolic extents drive clone-local masks and main-module output slices.
 *
 * Supports multiple simultaneous symbolic dimensions. `max_clones` limits the
 * cartesian product; zero disables the limit.
 */
struct MIGRAPHX_EXPORT split_sym_dim
{
    std::size_t max_clones = 64;

    std::string name() const { return "split_sym_dim"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
