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

    // ---------------------------------------------------------------------------
    // Dump flags — each causes the program to be printed to stderr after the
    // corresponding stage completes.  These are orthogonal and can be combined.
    // ---------------------------------------------------------------------------

    /// Dump the MIGraphX op graph immediately after DxGML parsing (before any
    /// lowering or optimization passes).
    bool dump_migraphx_ops = false;

    /// Dump the MIGraphX MLIR-dialect representation after it has been generated
    /// by the rocMLIR dialect lowering pass (requires the GPU target to be set).
    bool dump_migraphx_dialect = false;

    /// Dump the final GPU-lowered program (after program::compile() completes).
    bool dump_gpu = false;

    /// Dump the ISA / device assembly produced by the GPU back-end.
    /// On AMD hardware this is the GCN/RDNA assembly embedded in the code object.
    bool dump_isa = false;

    /// Path to a companion resources file (e.g. resources.mlir) that contains a
    /// {-# dialect_resources: { dxgml: { NAME: "0xHEX..." } } #-} block with
    /// weight tensor data.  When set, dxgml_op.constant operands whose resource
    /// name is found in the file are resolved to literal values instead of
    /// named parameter inputs.  Ignored when empty.
    std::string resources_file;
};

/// Parse a DxGML MLIR dialect file (.mlir) and return a MIGraphX program.
AMDXGML_EXPORT program parse_dxgml(const std::string& filename,
                                           const dxgml_options& options = {});

/// Parse DxGML MLIR dialect text and return a MIGraphX program.
AMDXGML_EXPORT program parse_dxgml_string(const std::string& mlir_text,
                                                   const dxgml_options& options = {});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_DXGML_HPP
