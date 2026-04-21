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
static const char* const winograd_kernel_src = R"__migraphx__(
#include <migraphx/kernels/winograd.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void winograd_kernel(void* x_p, void* w_p, void* y_p)
{
    make_tensors()(x_p, w_p, y_p)([](auto x, auto w, auto y) {
        winograd_conv_f2x3_s1_mn<${k_per_block}, ${tiles_per_block}, ${op_m}, ${op_n}>(x, w, y);
    });
}

}

} // namespace migraphx

)__migraphx__";

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

        // Output shape is the last input (after allocation injection).
        const auto& y_shape = inputs.back();
        const auto& y_lens  = y_shape.lens();
        const std::size_t N = y_lens.at(0);
        const std::size_t K = y_lens.at(1);
        const std::size_t H = y_lens.at(2);
        const std::size_t W = y_lens.at(3);

        const std::size_t tiles_h        = (H + 1) / 2;
        const std::size_t tiles_w        = (W + 1) / 2;
        const std::size_t tiles_per_img  = tiles_h * tiles_w;
        const std::size_t total_tiles    = N * tiles_per_img;

        // Tuning parameters.
        std::size_t k_per_block     = v.get("k_per_block", std::size_t{32});
        std::size_t tiles_per_block = v.get("tiles_per_block", std::size_t{32});
        std::size_t op_m            = v.get("op_m", std::size_t{2});
        std::size_t op_n            = v.get("op_n", std::size_t{2});

        if(op_m == 0)
            op_m = 1;
        if(op_n == 0)
            op_n = 1;

        // Ensure K_PER_BLOCK / TILES_PER_BLOCK are multiples of op_m/op_n.
        auto align = [](std::size_t v_, std::size_t a) {
            return ((v_ + a - 1) / a) * a;
        };
        k_per_block     = align(k_per_block, op_m);
        tiles_per_block = align(tiles_per_block, op_n);

        if(k_per_block > K)
            k_per_block = align(K, op_m);
        if(tiles_per_block > total_tiles)
            tiles_per_block = align(total_tiles, op_n);
        if(k_per_block == 0)
            k_per_block = op_m;
        if(tiles_per_block == 0)
            tiles_per_block = op_n;

        const std::size_t block_size = (k_per_block / op_m) * (tiles_per_block / op_n);

        const std::size_t num_k_blocks = (K + k_per_block - 1) / k_per_block;
        const std::size_t num_t_blocks = (total_tiles + tiles_per_block - 1) / tiles_per_block;
        const std::size_t num_blocks   = num_k_blocks * num_t_blocks;

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src = interpolate_string(winograd_kernel_src,
                                      {{"k_per_block", std::to_string(k_per_block)},
                                       {"tiles_per_block", std::to_string(tiles_per_block)},
                                       {"op_m", std::to_string(op_m)},
                                       {"op_n", std::to_string(op_n)}});

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
        auto shapes = to_shapes(ins->inputs());
        tc.problem  = to_value(shapes);
        const std::size_t wave =
            ctx.get_current_device().get_wavefront_size();
        const std::size_t max_block = 1024;

        auto add = [&](std::size_t kb, std::size_t tb, std::size_t om, std::size_t on) {
            if(kb % om != 0 or tb % on != 0)
                return;
            const auto t_k     = kb / om;
            const auto t_t     = tb / on;
            const auto block   = t_k * t_t;
            if(block < wave or block > max_block)
                return;
            if((block % wave) != 0)
                return;
            tc.solutions.push_back({{"k_per_block", kb},
                                    {"tiles_per_block", tb},
                                    {"op_m", om},
                                    {"op_n", on}});
        };

        if(exhaustive)
        {
            for(auto kb : {2, 4, 8, 16, 32, 64})
                for(auto tb : {2, 4, 8, 16, 32, 64, 128})
                    for(auto om : {1, 2, 4})
                        for(auto on : {1, 2, 4})
                            add(kb, tb, om, on);
        }
        else
        {
            // Ordered roughly by expected performance for medium/large problems.
            add(32, 32, 2, 2);
            add(16, 32, 2, 2);
            add(32, 16, 2, 2);
            add(16, 16, 2, 2);
            add(64, 16, 4, 2);
            add(16, 64, 2, 4);
            add(64, 32, 4, 2);
            add(32, 64, 2, 4);
            add(64, 64, 4, 4);
            add(32, 32, 1, 2);
            add(32, 32, 2, 1);
            add(16, 16, 1, 1);
            // Fallbacks for tiny problems where total tiles or K is small.
            add(8, 8, 1, 1);
            add(4, 16, 1, 1);
            add(16, 4, 1, 1);
            add(2, 32, 1, 1);
            add(32, 2, 1, 1);
            add(8, 16, 1, 1);
            add(16, 8, 1, 1);
            add(4, 32, 1, 1);
            add(32, 4, 1, 1);
            add(8, 32, 1, 1);
            add(32, 8, 1, 1);
        }
        if(tc.solutions.empty())
            tc.solutions.push_back(
                {{"k_per_block", 16}, {"tiles_per_block", 16}, {"op_m", 1}, {"op_n", 1}});
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
