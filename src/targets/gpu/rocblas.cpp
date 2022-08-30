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

#include <unordered_set>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

rocblas_handle_ptr create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return rocblas_handle_ptr{handle};
}

rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s)
{
    rocblas_handle_ptr rb = create_rocblas_handle_ptr();
    rocblas_set_stream(rb.get(), s);
    return rb;
}

const std::unordered_set<std::string>& get_rocblas_fp32_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx908", "gfx90a"};
    return supported_archs;
}

bool get_compute_fp32_flag()
{
    bool compute_fp32 = false;
#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    if(contains(get_rocblas_fp32_archs(), device_name))
        compute_fp32 = true;
#endif
    return compute_fp32;
}

bool get_int8_x4_format(context& ctx)
{
    bool int8_x4_format = true;
#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
    rocblas_gemm_flags flag;
    rocblas_query_int8_layout_flag(ctx.get_stream().get_rocblas(), &flag);
    int8_x4_format = (flag == rocblas_gemm_flags_pack_int8x4);
#endif
    return int8_x4_format;
}
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
