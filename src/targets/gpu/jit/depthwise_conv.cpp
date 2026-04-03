/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Optimized depthwise convolution kernel using shared memory + register tiling
// Each thread computes REG_X outputs in X direction
// Block covers (TILE_W * REG_X) × TILE_H outputs
static const char* const depthwise_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>

namespace migraphx {

extern "C" {

__global__
__attribute__((amdgpu_flat_work_group_size(${block_size}, ${block_size})))
void depthwise_conv_kernel(const float* __restrict__ input,
                           const float* __restrict__ weight,
                           ${bias_param}
                           float* __restrict__ output)
{
    constexpr int N = ${batch};
    constexpr int C = ${channels};
    constexpr int H = ${height};
    constexpr int W = ${width};
    constexpr int KH = ${kernel_h};
    constexpr int KW = ${kernel_w};
    constexpr int padH = ${pad_h};
    constexpr int padW = ${pad_w};
    // Note: stride is assumed to be 1 (checked in is_depthwise_conv matcher)
    constexpr int outH = ${out_h};
    constexpr int outW = ${out_w};

    // 32 x 8 tilling has better perf than 16 x 16
    constexpr int TILE_W = 32;
    constexpr int TILE_H = 8;
    constexpr int REG_X = 2;
    constexpr int OUT_TILE_W = TILE_W * REG_X;
    constexpr int SMEM_H = TILE_H + KH - 1;
    constexpr int SMEM_W = OUT_TILE_W + KW - 1;

    // Migraphx has 1D grid launch
    constexpr int GRID_X = ${grid_x};
    constexpr int GRID_Y = ${grid_y};

    __shared__ float s_input[SMEM_H][SMEM_W];

    const int tid = threadIdx.x;
    const int tx = tid % TILE_W;  // 0..31
    const int ty = tid / TILE_W;  // 0..7

    const int block_id = blockIdx.x;
    const int bx = block_id % GRID_X;
    const int by = (block_id / GRID_X) % GRID_Y;
    const int bz = block_id / (GRID_X * GRID_Y);

    const int c = bz % C;
    const int n = bz / C;

    const int out_x_base = bx * OUT_TILE_W + tx * REG_X;
    const int out_y = by * TILE_H + ty;

    // Filter in registers has better perf than Share memory
    float w_reg[KH * KW];
    #pragma unroll
    for (int i = 0; i < KH * KW; i++) {
        w_reg[i] = weight[c * KH * KW + i];
    }

    const int in_x_start = bx * OUT_TILE_W - padW;
    const int in_y_start = by * TILE_H - padH;

    const int smem_size = SMEM_H * SMEM_W;
    for (int i = tid; i < smem_size; i += blockDim.x) {
        int dy = i / SMEM_W;
        int dx = i % SMEM_W;
        int in_y = in_y_start + dy;
        int in_x = in_x_start + dx;

        float val = 0.0f;
        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
            val = input[((n * C + c) * H + in_y) * W + in_x];
        }
        s_input[dy][dx] = val;
    }
    __syncthreads();

    if (out_y >= outH || n >= N) return;

    float sum0 = 0.0f, sum1 = 0.0f;
    const int sx = tx * REG_X;

    #pragma unroll
    for (int kh = 0; kh < KH; kh++) {
        #pragma unroll
        for (int kw = 0; kw < KW; kw++) {
            float wv = w_reg[kh * KW + kw];
            sum0 += s_input[ty + kh][sx + kw] * wv;
            sum1 += s_input[ty + kh][sx + 1 + kw] * wv;
        }
    }

    ${bias_add}

    int ox0 = out_x_base;
    if (ox0 < outW)
        output[((n * C + c) * outH + out_y) * outW + ox0] = sum0;
    int ox1 = out_x_base + 1;
    if (ox1 < outW)
        output[((n * C + c) * outH + out_y) * outW + ox1] = sum1;
}

}

} // namespace migraphx
)__migraphx__";

struct depthwise_conv_compiler : compiler<depthwise_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::depthwise_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        bool has_bias = v.get("has_bias", false);
        auto input_shape  = inputs[0];
        auto weight_shape = inputs[1];
        auto padding_vec = v.at("padding").to_vector<std::size_t>();
        auto n     = input_shape.lens()[0];
        auto c     = input_shape.lens()[1];
        auto h     = input_shape.lens()[2];
        auto w     = input_shape.lens()[3];
        auto kh    = weight_shape.lens()[2];
        auto kw    = weight_shape.lens()[3];

        auto pad_h = padding_vec.size() >= 1 ? padding_vec[0] : 0;
        auto pad_w = padding_vec.size() >= 2 ? padding_vec[1] : 0;
        auto out_h = h + 2 * pad_h - kh + 1;
        auto out_w = w + 2 * pad_w - kw + 1;
        shape output_shape{input_shape.type(), {n, c, out_h, out_w}};

        constexpr int TILE_W = 32;
        constexpr int TILE_H = 8;
        constexpr int REG_X = 2;
        constexpr int OUT_TILE_W = TILE_W * REG_X;

        auto grid_x = (out_w + OUT_TILE_W - 1) / OUT_TILE_W;
        auto grid_y = (out_h + TILE_H - 1) / TILE_H;
        auto grid_z = n * c;
        auto block_size = TILE_W * TILE_H;

        hip_compile_options options;
        options.inputs = inputs;
        options.output = output_shape;
        options.kernel_name = "depthwise_conv_kernel";

        options.set_launch_params(v, grid_x * grid_y * grid_z * block_size, block_size);

        std::string bias_param = has_bias ? "const float* __restrict__ bias," : "";
        std::string bias_add = has_bias ?
            "{ float b = bias[c]; sum0 += b; sum1 += b; }" : "";

        auto src = interpolate_string(depthwise_conv_kernel,
                                      {{"batch", std::to_string(n)},
                                       {"channels", std::to_string(c)},
                                       {"height", std::to_string(h)},
                                       {"width", std::to_string(w)},
                                       {"kernel_h", std::to_string(kh)},
                                       {"kernel_w", std::to_string(kw)},
                                       {"pad_h", std::to_string(pad_h)},
                                       {"pad_w", std::to_string(pad_w)},
                                       {"out_h", std::to_string(out_h)},
                                       {"out_w", std::to_string(out_w)},
                                       {"grid_x", std::to_string(grid_x)},
                                       {"grid_y", std::to_string(grid_y)},
                                       {"block_size", std::to_string(block_size)},
                                       {"bias_param", bias_param},
                                       {"bias_add", bias_add}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v = op.to_value();
        auto result = compile_op(ctx, shapes, v);
        return result;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
