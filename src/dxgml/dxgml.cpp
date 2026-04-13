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
#include "dxgml_parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

program parse_dxgml(const std::string& filename, const dxgml_options& options)
{
    std::ifstream f(filename);
    if(!f)
        MIGRAPHX_THROW("DxGML: cannot open file: " + filename);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_dxgml_string(ss.str(), options);
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

    if(options.dump_migraphx_ops)
        std::cerr << "[DxGML] MIGraphX ops after parsing:\n" << parser.prog << "\n";

    // dump_migraphx_dialect, dump_gpu, and dump_isa require compilation and
    // are applied by the caller after program::compile() — they are stored in
    // dxgml_options so that tooling wrappers can inspect and act on them.

    return std::move(parser.prog);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
