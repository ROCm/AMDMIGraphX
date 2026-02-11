/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP

#include <migraphx/config.hpp>
#include <migraphx/tracer.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct mlir_ops_options
{
#ifdef _WIN32
#ifdef MIGRAPHX_USE_MIOPEN
    bool convolution = false;
    bool convolution_backwards = false;
    bool fused_convolution = false;
#else
    bool convolution = true;
    bool convolution_backwards = true;
    bool fused_convolution = true;
#endif
#ifdef MIGRAPHX_USE_HIPBLASLT
    bool attention = false;
    bool dot = false;
    bool fused_dot = false; 
#else
    bool attention = true;
    bool dot = true;
    bool fused_dot = true; 
#endif

#else
    bool convolution = false;
    bool convolution_backwards = false;
    bool fused_convolution = false;
    bool attention = false;
    bool dot = false;
    bool fused_dot = false;    
#endif
};

struct compile_options
{
    /**
     * Have MIGX allocate memory for parameters and add instructions
     * to copy parameters and output to/from an offload device like a GPU.
     */
    bool offload_copy = false;

    bool fast_math       = true;
    bool exhaustive_tune = false;

    mlir_ops_options mlir_ops{};

    tracer trace{};
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
