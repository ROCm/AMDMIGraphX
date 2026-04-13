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

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Internal DxGML MLIR-to-MIGraphX parser.
///
/// Implements a hand-rolled text parser for DxGML MLIR files.  No MLIR C API
/// or dialect registration is required — all parsing is done directly from the
/// printed text using the well-defined DxGML grammar subset.
struct dxgml_parser
{
    dxgml_options opts;
    program prog;
    module* mm = nullptr;

    /// SSA name → instruction_ref (populated as we walk ops).
    std::unordered_map<std::string, instruction_ref> value_map;

    /// Parse DxGML MLIR text and populate `prog`.
    void parse_from_string(const std::string& mlir_text);

    // Helpers used by parse_ops.cpp
    shape parse_tensor_type(const std::string& type_str) const;
    shape::type_t parse_element_type(const std::string& elem_str) const;
    std::vector<std::size_t> parse_dense_int_vec(const std::string& attr_str) const;
    int64_t parse_int_scalar(const std::string& attr_str) const;
    double parse_float_scalar(const std::string& attr_str) const;
    std::string get_attr_str(const std::string& attrs_block, const std::string& key) const;

    private:
    void parse_entry_point(const std::string& sig_line, const std::string& body);
    void parse_op_line(const std::string& result_name, const std::string& op_name,
                       const std::string& operands_str, const std::string& attrs_block,
                       const std::string& type_sig);

    instruction_ref parse_dxgml_op(const std::string& name,
                                   const std::string& operands_str,
                                   const std::string& attrs_block,
                                   const std::string& type_sig,
                                   const std::string& result_base_name = {},
                                   int num_results = 1);
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_DXGML_PARSER_HPP
