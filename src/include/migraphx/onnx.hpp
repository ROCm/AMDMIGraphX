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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// struct to pass in onnx options to parser
struct onnx_options
{
    /// default batch size to use (if not specified in onnx file)
    std::size_t default_dim_value = 1;
    /// Explicitly specify the dims of an input
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims = {};
    /// Continue parsing onnx file if an unknown operator is found
    bool skip_unknown_operators = false;
    /// Print program if an error occurs
    bool print_program_on_error = false;
    /// Max iter num for the loop operator
    int64_t max_loop_iterations = 10;
};

/// Create a program from an onnx file
program parse_onnx(const std::string& name, const onnx_options& = onnx_options{});

/// Create a program from an onnx buffer
program parse_onnx_buffer(const std::string& buffer, const onnx_options& options);

/// Create a program from an onnx buffer
program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options);

std::vector<std::string> get_onnx_operators();

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
