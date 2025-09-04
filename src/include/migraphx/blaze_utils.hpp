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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_BLAZE_UTILS_HPP
#define MIGRAPHX_GUARD_BLAZE_UTILS_HPP

#include <migraphx/config.h>

#if MIGRAPHX_USE_BLAZE

#include <blaze/Blaze.h>
#include <migraphx/tensor_view.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Utility functions for integrating Blaze library with MIGraphX
 */
namespace blaze_utils {

/**
 * @brief Convert a MIGraphX tensor_view to a Blaze DynamicMatrix
 * @tparam T Element type
 * @param tv MIGraphX tensor_view (must be 2D)
 * @return Blaze DynamicMatrix view of the tensor data
 */
template<typename T>
auto to_blaze_matrix(const tensor_view<T>& tv) 
{
    const auto& s = tv.get_shape();
    if(s.ndim() != 2)
        MIGRAPHX_THROW("Blaze matrix conversion requires 2D tensor");
    
    // Create a Blaze CustomMatrix that wraps the MIGraphX tensor data
    // Note: This creates a view, not a copy - modifications will affect original data
    return blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded, blaze::rowMajor>(
        tv.data(), s.lens()[0], s.lens()[1], s.strides()[1]);
}

/**
 * @brief Convert a MIGraphX tensor_view to a Blaze DynamicVector
 * @tparam T Element type
 * @param tv MIGraphX tensor_view (must be 1D)
 * @return Blaze DynamicVector view of the tensor data
 */
template<typename T>
auto to_blaze_vector(const tensor_view<T>& tv)
{
    const auto& s = tv.get_shape();
    if(s.ndim() != 1)
        MIGRAPHX_THROW("Blaze vector conversion requires 1D tensor");
    
    // Create a Blaze CustomVector that wraps the MIGraphX tensor data
    return blaze::CustomVector<T, blaze::unaligned, blaze::unpadded>(
        tv.data(), s.lens()[0], s.strides()[0]);
}

/**
 * @brief Perform optimized matrix multiplication using Blaze
 * @tparam T Element type
 * @param c Output tensor_view (2D)
 * @param a Input tensor_view A (2D)
 * @param b Input tensor_view B (2D)
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for existing C values
 */
template<typename T>
void blaze_gemm(tensor_view<T> c, 
                const tensor_view<T>& a, 
                const tensor_view<T>& b,
                T alpha = T{1}, 
                T beta = T{0})
{
    auto blaze_c = to_blaze_matrix(c);
    auto blaze_a = to_blaze_matrix(a);
    auto blaze_b = to_blaze_matrix(b);
    
    if(beta == T{0})
    {
        // C = alpha * A * B
        blaze_c = alpha * (blaze_a * blaze_b);
    }
    else
    {
        // C = alpha * A * B + beta * C
        blaze_c = alpha * (blaze_a * blaze_b) + beta * blaze_c;
    }
}

/**
 * @brief Check if Blaze library is available and configured
 * @return true if Blaze support is enabled, false otherwise
 */
constexpr bool is_blaze_available() { return true; }

} // namespace blaze_utils
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#else // MIGRAPHX_USE_BLAZE

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace blaze_utils {

/**
 * @brief Check if Blaze library is available and configured
 * @return false when Blaze support is disabled
 */
constexpr bool is_blaze_available() { return false; }

} // namespace blaze_utils
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_USE_BLAZE

#endif // MIGRAPHX_GUARD_BLAZE_UTILS_HPP
