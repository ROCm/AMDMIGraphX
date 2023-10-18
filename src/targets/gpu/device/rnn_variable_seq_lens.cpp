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
#include <migraphx/gpu/device/rnn_variable_seq_lens.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_var_sl_shift_sequence(hipStream_t stream,
                               const argument& result,
                               const argument& arg_hs,
                               const argument& arg_sl,
                               int layout)
{
    auto output_shape = result.get_shape();
    // layout = 0 [seq_length, batch_size, input_size]
    // layout = 1 [batch_size, seq_length, input_size]
    int seq_index   = (layout == 0) ? 0 : 1;
    int batch_index = (layout == 0) ? 1 : 0;
    int64_t max_len = output_shape.lens()[seq_index];
    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        auto out_s          = make_hip_shape<3>(output_shape);
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            gs_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto t   = idx[seq_index];
                auto b   = idx[batch_index];
                auto l   = sl_data[b];
                auto val = in_data[0];
                val      = 0;
                if(t >= max_len - l)
                {
                    auto in_idx = idx;
                    in_idx[seq_index] -= (max_len - l);
                    val = in_data[out_s.index(in_idx)];
                }
                out_data[i] = val;
            });
        });
    });
}

void rnn_var_sl_shift_output(hipStream_t stream,
                             const argument& result,
                             const argument& arg_hs,
                             const argument& arg_sl,
                             bool is_reverse,
                             int layout)
{
    auto output_shape = result.get_shape();
    // layout = 0 [seq_length, num_directions, batch_size, hidden_size]
    // layout = 1 [batch_size, seq_length, num_directions, hidden_size]
    int seq_index   = (layout == 0) ? 0 : 1;
    int batch_index = (layout == 0) ? 2 : 0;
    int64_t max_len = output_shape.lens()[seq_index];
    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        auto out_s          = make_hip_shape<4>(output_shape);
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            gs_launch(stream, output_shape.elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto t   = idx[seq_index];
                auto d   = idx[seq_index + 1];
                auto b   = idx[batch_index];
                auto l   = sl_data[b];
                auto val = in_data[0];
                val      = 0;
                if(t < l)
                {
                    int offset  = (d == 1 or is_reverse) ? 1 : 0;
                    auto in_idx = idx;
                    in_idx[seq_index] += offset * (max_len - l);
                    val = in_data[out_s.index(in_idx)];
                }
                out_data[i] = val;
            });
        });
    });
}

void rnn_var_sl_last_output(hipStream_t stream,
                            const argument& result,
                            const argument& arg_hs,
                            const argument& arg_sl,
                            bool is_reverse,
                            int layout)
{
    auto input_shape   = arg_hs.get_shape();
    auto out_comp_lens = input_shape.lens();
    // layout = 0 [seq_length, num_directions, batch_size, hidden_size]
    // layout = 1 [batch_size, seq_length, num_directions, hidden_size]
    int seq_index            = (layout == 0) ? 0 : 1;
    int batch_index          = (layout == 0) ? 2 : 0;
    out_comp_lens[seq_index] = 1;
    shape out_comp_shape{input_shape.type(), out_comp_lens};

    visit_all(result, arg_hs)([&](auto output, auto input) {
        const auto* in_data = device_cast(input.data());
        auto* out_data      = device_cast(output.data());
        arg_sl.visit([&](auto sl) {
            const auto* sl_data = device_cast(sl.data());
            auto in_s           = make_hip_shape<4>(input_shape);
            auto out_s          = make_hip_shape<4>(out_comp_shape);
            gs_launch(stream, result.get_shape().elements(), 256)([=](auto i) __device__ {
                auto idx = out_s.multi(i);
                auto d   = idx[seq_index + 1];
                auto b   = idx[batch_index];
                auto l   = sl_data[b];
                if(is_reverse or d == 1)
                {
                    idx[seq_index] = 0;
                }
                else
                {
                    idx[seq_index] = l - 1;
                }
                out_data[i] = in_data[in_s.index(idx)];
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
