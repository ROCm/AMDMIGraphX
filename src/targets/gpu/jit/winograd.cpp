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

        std::size_t batch       = in_lens[0];
        std::size_t channels    = in_lens[1];
        std::size_t height      = in_lens[2];
        std::size_t width       = in_lens[3];
        std::size_t filters     = w_lens[0];
        int group               = v.at("group").to<int>();
        bool pretransformed     = v.get("pretransformed", false);
        std::size_t cpg         = channels / group;

        // F(2x2, 3x3)
        std::size_t tiles_h     = (height + 1) / 2;
        std::size_t tiles_w     = (width + 1) / 2;
        std::size_t total_tiles = tiles_h * tiles_w;

        // Read tuning parameters from solution, with heuristic defaults
        std::size_t block_size =
            v.get("block_size", (total_tiles <= 64) ? std::size_t{128} : std::size_t{256});
        std::size_t k_per_wg = v.get("k_per_wg", std::min(filters, std::size_t{32}));
        k_per_wg             = (k_per_wg / 2) * 2;
        if(k_per_wg == 0)
            k_per_wg = 2;

        // T_TILE=2, K_TILE=2 → tiles_per_wg*k_per_wg = 4*block_size
        std::size_t tiles_per_wg = 4 * block_size / k_per_wg;

        // CHUNK_C: max channels per LDS batch (64KB LDS limit)
        std::size_t max_lds = 65536;
        std::size_t chunk_c = max_lds / (16 * (tiles_per_wg + k_per_wg) * sizeof(float));
        chunk_c             = std::min(chunk_c, cpg);
        if(chunk_c == 0)
            chunk_c = 1;

        std::size_t lds_floats = 16 * (tiles_per_wg * chunk_c + chunk_c * k_per_wg);

        std::size_t tile_groups = (total_tiles + tiles_per_wg - 1) / tiles_per_wg;
        std::size_t k_groups    = (filters + k_per_wg - 1) / k_per_wg;
        std::size_t total_wgs   = batch * tile_groups * k_groups;

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "winograd_kernel";
        options.set_launch_params(v, total_wgs * block_size, block_size);

        auto src = interpolate_string(winograd_kernel,
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

    // 4-arg compile: receives solution from tuning system
    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        // Merge solution params into op values so compile_op sees them
        auto v = op.to_value();
        if(not solution.empty())
        {
            // Copy solution keys into v
            if(solution.contains("block_size"))
                v["block_size"] = solution.at("block_size");
            if(solution.contains("k_per_wg"))
                v["k_per_wg"] = solution.at("k_per_wg");
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config>
    get_tuning_config(context&, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        if(not exhaustive)
            return nullopt;

        auto shapes = to_shapes(ins->inputs());
        auto v      = op.to_value();

        auto in_lens        = shapes[0].lens();
        auto w_lens         = shapes[1].lens();
        std::size_t height  = in_lens[2];
        std::size_t width   = in_lens[3];
        std::size_t filters = w_lens[0];
        int group           = v.at("group").to<int>();
        std::size_t cpg     = in_lens[1] / group;

        std::size_t tiles_h     = (height + 1) / 2;
        std::size_t tiles_w     = (width + 1) / 2;
        std::size_t total_tiles = tiles_h * tiles_w;

        tuning_config tc;
        tc.problem = to_value(shapes);

        // Generate candidate combinations of (block_size, k_per_wg)
        for(std::size_t bs : {64, 128, 256})
        {
            for(std::size_t kpw : {4, 8, 16, 32})
            {
                if(kpw > filters)
                    continue;
                if(filters % kpw != 0)
                    continue;
                std::size_t tpw = 4 * bs / kpw;
                // Check LDS fits
                std::size_t max_chunk =
                    65536 / (16 * (tpw + kpw) * sizeof(float));
                if(max_chunk == 0)
                    continue;
                std::size_t cc = std::min(max_chunk, cpg);
                if(cc == 0)
                    continue;

                tc.solutions.push_back(
                    {{"block_size", bs}, {"k_per_wg", kpw}});
            }
        }

        // Always include a default solution
        if(tc.solutions.empty())
        {
            tc.solutions.push_back(value{});
        }

        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
