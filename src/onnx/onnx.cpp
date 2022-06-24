/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/op_parser.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <array>
#include <utility>
#include <vector>

#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class... Ts>
program parse_onnx_from(const onnx_options& options, Ts&&... xs)
{
    onnx::onnx_parser parser;
    parser.map_input_dims         = options.map_input_dims;
    parser.default_dim_value      = options.default_dim_value;
    parser.skip_unknown_operators = options.skip_unknown_operators;
    parser.max_loop_iterations    = options.max_loop_iterations;

    if(options.print_program_on_error)
    {
        // Log the program when it can't be parsed
        try
        {
            parser.parse_from(std::forward<Ts>(xs)...);
        }
        catch(...)
        {
            std::cerr << parser.prog << std::endl;
            throw;
        }
    }
    else
    {
        parser.parse_from(std::forward<Ts>(xs)...);
    }
    return std::move(parser.prog);
}

program parse_onnx(const std::string& name, const onnx_options& options)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    return parse_onnx_from(options, input, name);
}

program parse_onnx_buffer(const std::string& buffer, const onnx_options& options)
{
    return parse_onnx_from(options, buffer.data(), buffer.size());
}

program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options)
{
    return parse_onnx_from(options, data, size);
}

std::vector<std::string> get_onnx_operators() { return onnx::get_op_parsers(); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
