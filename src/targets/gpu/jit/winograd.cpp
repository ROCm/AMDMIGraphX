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
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const winograd_kernel = R"__migraphx__(
#include <migraphx/kernels/winograd.hpp>
#include <migraphx/kernels/index.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void winograd_kernel(void* input_ptr, void* weight_ptr, void* output_ptr)
{
    __shared__ float s_filt[${k_batch} * ${chunk_c} * 36];
    const auto* input  = static_cast<const float*>(input_ptr);
    const auto* weight = static_cast<const float*>(weight_ptr);
    auto* output       = static_cast<float*>(output_ptr);
    winograd::conv<${group}, ${batch}, ${channels}, ${height}, ${width},
                   ${filters}, ${chunk_c}, ${k_batch}>(
        input, weight, output, s_filt);
}

}

} // namespace migraphx

)__migraphx__";

struct winograd_compiler : compiler<winograd_compiler>
{
    std::vector<std::string> names() const { return {"gpu::pre_winograd_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& in_shape  = inputs[0];
        const auto& w_shape   = inputs[1];
        const auto& out_shape = inputs.back();

        auto in_lens = in_shape.lens();
        auto w_lens  = w_shape.lens();

        std::size_t batch       = in_lens[0];
        std::size_t channels    = in_lens[1];
        std::size_t height      = in_lens[2];
        std::size_t width       = in_lens[3];
        std::size_t filters     = w_lens[0];
        int group               = v.at("group").to<int>();
        std::size_t c_per_group = channels / group;

        std::size_t tiles_h   = (height + 3) / 4;
        std::size_t tiles_w   = (width + 3) / 4;
        std::size_t num_tiles = tiles_h * tiles_w;

        // =====================================================================
        // Tuning parameters
        // =====================================================================

        // K_BATCH: output filters per thread. More = less memory traffic but
        // more VGPRs. Budget: K_BATCH*36 + ~60 overhead must fit in 256 VGPRs.
        // K_BATCH=4 → 204 VGPRs (1 wave), K_BATCH=2 → 132 VGPRs (2 waves).
        // For memory-bound workloads (large spatial), prefer K_BATCH=4.
        // For small spatial (few tiles), prefer K_BATCH=2 for occupancy.
        std::size_t k_batch = 4;
        if(num_tiles <= 64)
            k_batch = 2;
        if(k_batch > filters)
            k_batch = filters;
        // Ensure k_batch divides filters evenly, or find largest that does
        while(k_batch > 1 and filters % k_batch != 0)
            k_batch--;

        // CHUNK_C: channels per shared memory batch.
        // Shared memory = k_batch * chunk_c * 36 * 4 bytes. Cap at 32KB.
        std::size_t max_smem = 32768;
        std::size_t chunk_c  = max_smem / (k_batch * 36 * sizeof(float));
        if(chunk_c > c_per_group)
            chunk_c = c_per_group;
        if(chunk_c == 0)
            chunk_c = 1;

        // =====================================================================
        // Launch parameters
        // =====================================================================
        std::size_t block_size  = 256;
        std::size_t k_groups    = (filters + k_batch - 1) / k_batch;
        std::size_t tile_groups = (num_tiles + block_size - 1) / block_size;
        std::size_t total_wgs   = batch * tile_groups * k_groups;
        std::size_t global      = total_wgs * block_size;

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = out_shape;
        options.kernel_name = "winograd_kernel";
        options.set_launch_params(v, global, block_size);

        auto src = interpolate_string(winograd_kernel,
                                      {{"batch", to_string(batch)},
                                       {"channels", to_string(channels)},
                                       {"height", to_string(height)},
                                       {"width", to_string(width)},
                                       {"filters", to_string(filters)},
                                       {"group", to_string(group)},
                                       {"chunk_c", to_string(chunk_c)},
                                       {"k_batch", to_string(k_batch)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
