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
    __shared__ float lds[${lds_floats}];
    winograd::conv<${group}, ${batch}, ${channels}, ${height}, ${width},
                   ${filters}, ${tiles_per_wg}, ${k_per_wg}, ${chunk_c},
                   ${pretransformed}>(
        static_cast<const float*>(input_ptr),
        static_cast<const float*>(weight_ptr),
        static_cast<float*>(output_ptr),
        lds);
}

}

} // namespace migraphx

)__migraphx__";

struct winograd_compiler : compiler<winograd_compiler>
{
    std::vector<std::string> names() const { return {"gpu::pre_winograd_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto in_lens = inputs[0].lens();
        auto w_lens  = inputs[1].lens();

        std::size_t batch   = in_lens[0];
        std::size_t channels = in_lens[1];
        std::size_t height  = in_lens[2];
        std::size_t width   = in_lens[3];
        std::size_t filters = w_lens[0];
        int group           = v.at("group").to<int>();
        bool pretransformed = v.get("pretransformed", false);
        std::size_t cpg     = channels / group;

        std::size_t tiles_h     = (height + 1) / 2;
        std::size_t tiles_w     = (width + 1) / 2;
        std::size_t total_tiles = tiles_h * tiles_w;

        // Tune block size: smaller blocks = more WGs for small spatial sizes
        std::size_t block_size = (total_tiles <= 64) ? 128 : 256;

        std::size_t k_per_wg = std::min(filters, std::size_t{32});
        k_per_wg             = (k_per_wg / 2) * 2;
        if(k_per_wg == 0)
            k_per_wg = 2;
        // T_TILE=2, K_TILE=2 → tiles_per_wg*k_per_wg = 4*block_size
        std::size_t tiles_per_wg = 4 * block_size / k_per_wg;
        std::size_t max_lds      = 65536;
        std::size_t chunk_c =
            max_lds / (16 * (tiles_per_wg + k_per_wg) * sizeof(float));
        chunk_c = std::min(chunk_c, cpg);
        if(chunk_c == 0)
            chunk_c = 1;
        std::size_t lds_floats =
            16 * (tiles_per_wg * chunk_c + chunk_c * k_per_wg);

        std::size_t tile_groups = (total_tiles + tiles_per_wg - 1) / tiles_per_wg;
        std::size_t k_groups    = (filters + k_per_wg - 1) / k_per_wg;
        std::size_t total_wgs   = batch * tile_groups * k_groups;

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "winograd_kernel";
        options.set_launch_params(v, total_wgs * block_size, block_size);

        auto src = interpolate_string(
            winograd_kernel,
            {{"group", to_string(group)},
             {"batch", to_string(batch)},
             {"channels", to_string(channels)},
             {"height", to_string(height)},
             {"width", to_string(width)},
             {"filters", to_string(filters)},
             {"tiles_per_wg", to_string(tiles_per_wg)},
             {"k_per_wg", to_string(k_per_wg)},
             {"chunk_c", to_string(chunk_c)},
             {"lds_floats", to_string(lds_floats)},
             {"pretransformed", pretransformed ? "true" : "false"}});

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
