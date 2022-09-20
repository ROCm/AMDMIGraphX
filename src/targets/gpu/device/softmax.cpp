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
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    auto batch_lens          = result.get_shape().lens();
    index_int batch_item_num = batch_lens[axis];
    batch_lens[axis]         = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {
        const index_int max_block_size = 128;
        const index_int block_size     = compute_block_size(batch_item_num, max_block_size);
        using type = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
        type init  = lowest();

        if(axis == batch_lens.size() - 1)
        {
            gs_launch(stream, batch_shape.elements() * block_size, block_size)(
                [=](auto i, auto idx) __device__ {
                    auto start_loc = i / block_size * batch_item_num;
                    auto batch_max = block_reduce<max_block_size>(
                        idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                            return input[start_loc + j];
                        });

                    auto batch_sum = block_reduce<max_block_size>(
                        idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                            auto val = input[start_loc + j] - batch_max;
                            return ::exp(to_hip_type(val));
                        });

                    idx.local_stride(batch_item_num, [&](auto j) __device__ {
                        auto val              = input[start_loc + j] - batch_max;
                        output[start_loc + j] = ::exp(to_hip_type(val)) / batch_sum;
                    });
                });
        }
        else
        {
            gs_launch(stream, batch_shape.elements() * block_size, block_size)(
                [=](auto i, auto idx) __device__ {
                    auto data_idx  = batch.multi(i / block_size);
                    auto batch_max = block_reduce<max_block_size>(
                        idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                            data_idx[axis] = j;
                            return input[data_idx];
                        });

                    auto batch_sum = block_reduce<max_block_size>(
                        idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                            data_idx[axis] = j;
                            auto val       = input[data_idx] - batch_max;
                            return ::exp(to_hip_type(val));
                        });

                    idx.local_stride(batch_item_num, [&](auto j) __device__ {
                        data_idx[axis]   = j;
                        auto val         = input[data_idx] - batch_max;
                        output[data_idx] = ::exp(to_hip_type(val)) / batch_sum;
                    });
                });
        }
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
