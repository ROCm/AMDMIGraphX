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
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/fixed_pad.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/functional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

static argument fixed_pad_base_impl(hipStream_t stream, const argument& result, const argument& arg)
{
    hip_visit_all(result, arg)([&](auto output, auto input) {
        gs_launch(stream, result.get_shape().elements())([=](auto i) __device__ {
            auto input_bounds = input.get_shape().lens;
            auto idx          = output.get_shape().multi(i);

            bool in_bounds = sequence(
                idx.size(), [&](auto... js) { return ((idx[js] < input_bounds[js]) and ...); });

            output[idx] = in_bounds ? input[idx] : 0;
        });
    });
    return result;
}

static argument
fixed_pad_standard_impl(hipStream_t stream, const argument& result, const argument& arg)
{
    index_int nelements = result.get_shape().elements();
    index_int ielements = arg.get_shape().elements();
    hip_pointer_visit_all(result, arg)([&](auto output, auto input) {
        gs_launch(stream, nelements)(
            [=](auto i) __device__ { output[i] = (i < ielements) ? input[i] : 0; });
    });
    return result;
}

argument fixed_pad(hipStream_t stream, const argument& result, const argument& arg)
{
    if(result.get_shape().standard() and arg.get_shape().standard())
    {
        auto ilens            = arg.get_shape().lens();
        auto olens            = result.get_shape().lens();
        auto [istart, ostart] = std::mismatch(ilens.begin(), ilens.end(), olens.begin());
        if(std::equal(istart, ilens.end(), ostart, olens.end()))
            return fixed_pad_standard_impl(stream, result, arg);
    }
    return fixed_pad_base_impl(stream, result, arg);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
