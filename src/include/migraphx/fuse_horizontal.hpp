/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_HORIZONTAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_HORIZONTAL_HPP

#include <migraphx/config.hpp>
#include <migraphx/pass_manager.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Horizontal fusion pass for GEMMs and pointwise operations.
 * 
 * This pass identifies groups of independent operations with compatible shapes
 * and fuses them into batched operations. This reduces kernel launch overhead
 * and can improve GPU utilization.
 * 
 * For dot operations: Multiple independent dots with the same shape are combined
 * into a single batched GEMM by concatenating inputs along a new batch dimension.
 * 
 * For pointwise operations: Multiple independent operations of the same type
 * and shape are combined similarly.
 */
struct MIGRAPHX_EXPORT fuse_horizontal
{
    std::string name() const { return "fuse_horizontal"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_HORIZONTAL_HPP






