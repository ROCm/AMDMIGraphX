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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_DXGML_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_DXGML_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dxgml/export.h>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Options controlling DxGML frontend behavior.
struct dxgml_options
{
    /// If true, unknown dxgml_op.* operators are silently skipped instead of throwing.
    bool skip_unknown_operators = false;
    /// If true, print the partial program to stderr when an error occurs.
    bool print_program_on_error = false;
};

/// Parse a DxGML MLIR dialect file (.mlir) and return a MIGraphX program.
MIGRAPHX_DXGML_EXPORT program parse_dxgml(const std::string& filename,
                                           const dxgml_options& options = {});

/// Parse DxGML MLIR dialect text and return a MIGraphX program.
MIGRAPHX_DXGML_EXPORT program parse_dxgml_string(const std::string& mlir_text,
                                                   const dxgml_options& options = {});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_DXGML_HPP
