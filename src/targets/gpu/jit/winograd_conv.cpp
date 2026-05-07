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

// Wave-distributed Winograd kernel. 16 lanes per element-group, each lane
// holds ONE Winograd element index e (0..15) and accumulates KT × TT
// (k, tile) outputs for that element. Mirrors MIOpen's gfx12 fp16_dot2
// element-distribution layout.
// NOLINTNEXTLINE
static const char* const winograd_kernel_src = R"__migraphx__(
#include <migraphx/kernels/winograd.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void winograd_kernel(void* x_p, void* w_p, void* y_p)
{
    make_tensors()(x_p, w_p, y_p)([](auto x, auto w, auto y) {
        winograd_conv_f2x3_s1_kernel<${kt_div},
                                     ${tt_div},
                                     ${kt},
                                     ${tt},
                                     ${ring}>(x, w, y);
    });
}

}

} // namespace migraphx

)__migraphx__";

// JIT compiler for gpu::winograd_conv. Tuning parameters mirror MIOpen's
// gfx12 fp16_dot2 Winograd kernel:
//
//   kt_div × tt_div   - lanes per element-group (block_size = 16 × kt_div × tt_div)
//   kt, tt            - per-thread (k, tile) accumulator tile for ITS element
//   ring              - LDS ring buffer depth (1, 2, or 4)
//
// MIOpen's canonical config: kt_div=4, tt_div=4, kt=8, tt=8 → 256-thread
// block, K_BLOCK=T_BLOCK=32, 64 fp32 acc per thread.
struct winograd_conv_compiler : compiler<winograd_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::winograd_conv", "winograd_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "winograd_kernel";
        options.virtual_inputs = inputs;

        const auto& y_shape = inputs.back();
        const auto& y_lens  = y_shape.lens();
        const std::size_t N = y_lens.at(0);
        const std::size_t K = y_lens.at(1);
        const std::size_t H = y_lens.at(2);
        const std::size_t W = y_lens.at(3);

        const std::size_t tiles_h       = (H + 1) / 2;
        const std::size_t tiles_w       = (W + 1) / 2;
        const std::size_t tiles_per_img = tiles_h * tiles_w;
        const std::size_t total_tiles   = N * tiles_per_img;

        std::size_t kt_div = v.get("kt_div", std::size_t{4});
        std::size_t tt_div = v.get("tt_div", std::size_t{4});
        std::size_t kt     = v.get("kt", std::size_t{8});
        std::size_t tt     = v.get("tt", std::size_t{8});
        std::size_t ring   = v.get("ring", std::size_t{1});

        if(kt == 0)
            kt = 1;
        if(tt == 0)
            tt = 1;
        if(kt_div == 0)
            kt_div = 1;
        if(tt_div == 0)
            tt_div = 1;
        if(ring < 1)
            ring = 1;
        if(ring > 4)
            ring = 4;

        const std::size_t k_block    = kt_div * kt;
        const std::size_t t_block    = tt_div * tt;
        const std::size_t block_size = 16u * kt_div * tt_div;

        // Constrain to actual problem.
        const std::size_t k_block_eff = std::min(k_block, K ? K : k_block);
        const std::size_t t_block_eff = std::min(t_block, total_tiles ? total_tiles : t_block);
        (void)k_block_eff;
        (void)t_block_eff;

        const std::size_t num_k_blocks = (K + k_block - 1) / k_block;
        const std::size_t num_t_blocks = (total_tiles + t_block - 1) / t_block;
        const std::size_t num_blocks   = num_k_blocks * num_t_blocks;

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src = interpolate_string(winograd_kernel_src,
                                      {{"kt_div", std::to_string(kt_div)},
                                       {"tt_div", std::to_string(tt_div)},
                                       {"kt", std::to_string(kt)},
                                       {"tt", std::to_string(tt)},
                                       {"ring", std::to_string(ring)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto v = op.to_value();
        for(const auto& s : solution)
            v.insert(s);
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation&,
                                              bool exhaustive) const
    {
        tuning_config tc;
        auto shapes                 = to_shapes(ins->inputs());
        tc.problem                  = to_value(shapes);
        const std::size_t wave      = ctx.get_current_device().get_wavefront_size();
        const std::size_t max_block = 1024;

        // Block_size = 16 × kt_div × tt_div must be a wavefront multiple
        // and within hardware limits.
        auto add = [&](std::size_t k_div,
                       std::size_t t_div,
                       std::size_t k_t,
                       std::size_t t_t,
                       std::size_t rg) {
            const auto block = 16u * k_div * t_div;
            if(block < wave or block > max_block)
                return;
            if((block % wave) != 0)
                return;
            tc.solutions.push_back(
                {{"kt_div", k_div}, {"tt_div", t_div}, {"kt", k_t}, {"tt", t_t}, {"ring", rg}});
        };

        if(exhaustive)
        {
            for(auto kd : {1, 2, 4, 8})
                for(auto td : {1, 2, 4, 8})
                    for(auto k_t : {1, 2, 4, 8, 16})
                        for(auto t_t : {1, 2, 4, 8, 16})
                            for(auto rg : {1, 2})
                                add(kd, td, k_t, t_t, rg);
        }
        else
        {
            // ---- High-occupancy variants (16 acc/thread, ~80 VGPR → 4 wg/CU).
            add(4, 4, 4, 4, 1); // 256-thr, 16x16 output
            add(4, 4, 4, 4, 2);
            add(2, 8, 4, 4, 1); // 256-thr, 8x32
            add(8, 2, 4, 4, 1); // 256-thr, 32x8
            add(2, 4, 4, 4, 1); // 128-thr, 8x16
            add(4, 2, 4, 4, 1); // 128-thr, 16x8
            add(2, 2, 4, 4, 1); // 64-thr, 8x8

            // ---- Medium tile (32 acc/thread, ~110 VGPR → 2 wg/CU).
            add(2, 4, 8, 4, 1);
            add(4, 2, 4, 8, 1);
            add(2, 2, 8, 4, 1);
            add(2, 2, 4, 8, 1);
            add(4, 4, 8, 4, 1);
            add(4, 4, 4, 8, 1);

            // ---- MIOpen-canonical 64 acc/thread (~120-200 VGPR → 1 wg/CU).
            add(4, 4, 8, 8, 1);
            add(4, 4, 8, 8, 2);
            add(2, 2, 8, 8, 1); // 64-thr block
            add(2, 2, 8, 8, 2);
            add(2, 8, 8, 8, 1); // 16x64 output
            add(2, 8, 8, 8, 2);
            add(8, 2, 8, 8, 1); // 64x16 output
            add(8, 2, 8, 8, 2);

            // ---- Wider per-thread K (winning some K-large cases).
            add(2, 8, 16, 8, 1);
            add(2, 8, 16, 8, 2);
            add(2, 4, 16, 8, 1);
            add(4, 2, 16, 8, 1);

            // ---- Tiny problems.
            add(2, 2, 2, 2, 1);
            add(1, 4, 4, 4, 1); // 64 threads
            add(4, 1, 4, 4, 1);
            add(1, 4, 2, 2, 1);
            add(4, 1, 2, 2, 1);
            add(2, 2, 2, 4, 1);
            add(2, 2, 4, 2, 1);
        }

        if(tc.solutions.empty())
            tc.solutions.push_back(
                {{"kt_div", 2}, {"tt_div", 2}, {"kt", 2}, {"tt", 2}, {"ring", 1}});
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
