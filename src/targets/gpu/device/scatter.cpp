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
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/scatter.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatter(
    hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2, int64_t axis)
{
    auto ds            = arg0.get_shape();
    auto inds          = arg1.get_shape();
    auto axis_dim_size = ds.lens()[axis];
    hip_visit_all(result, arg0, inds)([&](auto output, auto data, auto s1) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        gs_launch(stream, ds.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        hip_visit_all(arg1, arg2)([&](auto indices, auto update) {
            const auto* upd_ptr     = device_cast(update.data());
            const auto* indices_ptr = device_cast(indices.data());
            gs_launch(stream, inds.elements())([=](auto i) __device__ {
                auto out_idx    = s1.multi(i);
                auto index      = indices_ptr[i];
                index           = index < 0 ? index + axis_dim_size : index;
                out_idx[axis]   = index;
                output[out_idx] = upd_ptr[i];
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
