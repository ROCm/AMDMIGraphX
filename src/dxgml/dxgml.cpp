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
#include <migraphx/dxgml.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/fuse_dxgml_dequant.hpp>
#include <migraphx/fuse_dxgml_amdgpu_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include "dxgml_parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Convert a UTF-16 LE string (with or without BOM) to UTF-8.
// Only handles the Basic Multilingual Plane (code points < 0x10000),
// which covers all MLIR source text in practice.
static std::string utf16le_to_utf8(const std::string& raw)
{
    const auto* p   = reinterpret_cast<const unsigned char*>(raw.data());
    std::size_t len = raw.size();
    // Skip BOM (FF FE) if present
    std::size_t start = 0;
    if(len >= 2 && p[0] == 0xFF && p[1] == 0xFE)
        start = 2;

    std::string out;
    out.reserve(len); // upper bound
    for(std::size_t i = start; i + 1 < len; i += 2)
    {
        unsigned cp = static_cast<unsigned>(p[i]) | (static_cast<unsigned>(p[i + 1]) << 8);
        if(cp < 0x80)
        {
            out += static_cast<char>(cp);
        }
        else if(cp < 0x800)
        {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else
        {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

program parse_dxgml(const std::string& filename, const dxgml_options& options)
{
    std::ifstream f(filename, std::ios::binary);
    if(!f)
        MIGRAPHX_THROW("DxGML: cannot open file: " + filename);
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    // Detect UTF-16 LE BOM (FF FE) and convert to UTF-8
    const auto* raw = reinterpret_cast<const unsigned char*>(content.data());
    if(content.size() >= 2 && raw[0] == 0xFF && raw[1] == 0xFE)
        content = utf16le_to_utf8(content);

    return parse_dxgml_string(content, options);
}

program parse_dxgml_string(const std::string& mlir_text, const dxgml_options& options)
{
    dxgml_parser parser;
    parser.opts = options;

    if(options.print_program_on_error)
    {
        try
        {
            parser.parse_from_string(mlir_text);
        }
        catch(const std::exception& e)
        {
            std::cerr << "[DxGML] Parse error: " << e.what() << "\n";
            std::cerr << "[DxGML] Partial program:\n" << parser.prog << "\n";
            throw;
        }
        catch(...)
        {
            std::cerr << "[DxGML] Parse error (unknown exception)\n";
            std::cerr << "[DxGML] Partial program:\n" << parser.prog << "\n";
            throw;
        }
    }
    else
    {
        parser.parse_from_string(mlir_text);
    }

    // Apply DxGML-specific IR passes.
    std::vector<pass> dxgml_passes;
    dxgml_passes.push_back(fuse_dxgml_dequant{});
    if(not options.amdgpu_kernel_registry_file.empty())
        dxgml_passes.push_back(fuse_dxgml_amdgpu_ops{options.amdgpu_kernel_registry_file});
    dxgml_passes.push_back(dead_code_elimination{});
    migraphx::run_passes(*parser.prog.get_main_module(), dxgml_passes);

    if(options.dump_migraphx_ops)
        std::cerr << "[DxGML] MIGraphX ops after parsing:\n" << parser.prog << "\n";

    // dump_migraphx_dialect, dump_gpu, and dump_isa require compilation and
    // are applied by the caller after program::compile() — they are stored in
    // dxgml_options so that tooling wrappers can inspect and act on them.

    return std::move(parser.prog);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
