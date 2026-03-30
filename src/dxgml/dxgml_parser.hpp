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
#ifndef MIGRAPHX_GUARD_DXGML_PARSER_HPP
#define MIGRAPHX_GUARD_DXGML_PARSER_HPP

#include <migraphx/dxgml.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/shape.hpp>

// MLIR C API — no LLVM headers needed
#include <mlir-c/IR.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Internal DxGML MLIR-to-MIGraphX parser.
/// Uses the MLIR C API so it does not require DxGML or LLVM C++ headers.
/// Dialect registration is performed by DxgmlIRCreateContext() from dxgml_ir.dll.
struct dxgml_parser
{
    dxgml_options opts;
    program prog;
    module* mm = nullptr;

    /// SSA value id → instruction_ref, populated as we walk ops.
    std::unordered_map<intptr_t, instruction_ref> value_map;

    /// Parse DxGML MLIR text and populate `prog`.
    void parse_from_string(const std::string& mlir_text);

    // Accessible from parse_ops.cpp helpers
    shape mlir_type_to_shape(MlirType t) const;
    shape::type_t mlir_element_type_to_migraphx(MlirType elem_type) const;
    std::vector<std::size_t> get_dense_int_vec(MlirAttribute a) const;
    int64_t get_int_scalar(MlirAttribute a) const;

    private:
    void parse_entry_point(MlirOperation ep);
    void parse_op(MlirOperation op);

    instruction_ref parse_dxgml_op(const std::string& name,
                                   MlirOperation op,
                                   const std::vector<instruction_ref>& inputs);

    /// Stable integer id for an SSA value (use pointer as-is: values are unique).
    static intptr_t value_id(MlirValue v) { return reinterpret_cast<intptr_t>(v.ptr); }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_DXGML_PARSER_HPP
