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
static const char* const channelwise_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/channelwise_conv.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void channelwise_conv_kernel(void* x_p, void* w_p, void* y_p)
{
    transform_args(make_tensors(), rotate_last())(x_p, w_p, y_p)([](auto output, auto x, auto w) {
        channelwise_conv(index_ints<${tile}>{}, output, x, w);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct channelwise_conv_compiler : compiler<channelwise_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::channelwise_conv", "channelwise_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto num_spatial       = v.at("num_spatial").to<std::size_t>();
        const auto& x_s        = inputs.at(0);
        const auto& w_s        = inputs.at(1);
        const auto& out_s      = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "channelwise_conv_kernel";
        options.virtual_inputs = inputs;

        auto x_lens   = x_s.lens();
        auto w_lens   = w_s.lens();
        auto out_lens = out_s.lens();

        // Tile dimensions: for 2D use 8xH, 32xW; for 1D use 256
        std::vector<std::size_t> tile_sizes(num_spatial);
        if(num_spatial == 1)
        {
            tile_sizes[0] = 256;
        }
        else
        {
            tile_sizes[0]               = v.get("tile_h", 8);
            tile_sizes[num_spatial - 1] = v.get("tile_w", 32);
            for(std::size_t d = 1; d + 1 < num_spatial; ++d)
                tile_sizes[d] = 1;
        }

        std::size_t block_size = 1;
        for(auto t : tile_sizes)
            block_size *= t;

        // Compute number of tiles per spatial dim: ceil(out_spatial / tile)
        std::size_t num_blocks = out_lens[0] * out_lens[1];
        for(std::size_t d = 0; d < num_spatial; ++d)
        {
            auto out_spatial = out_lens[2 + d];
            num_blocks *= (out_spatial + tile_sizes[d] - 1) / tile_sizes[d];
        }

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src =
            interpolate_string(channelwise_conv_kernel, {{"tile", to_string_range(tile_sizes)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto v        = op.to_value();
        for(const auto& x : solution)
            v.insert(x);
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation& op,
                                              bool exhaustive) const
    {
        tuning_config tc;
        auto shapes       = to_shapes(ins->inputs());
        tc.problem        = to_value(shapes);
        if(exhaustive)
        {
            std::vector<std::size_t> sizes;
            for(auto i:range(1, 64))
                sizes.push_back(i*4);
            for(auto tile_h:sizes)
            {
                for(auto tile_w:sizes)
                {
                    auto block_size = tile_h * tile_w;
                    if(block_size > 1024)
                        continue;
                    if(block_size < ctx.get_current_device().get_wavefront_size())
                        continue;
                    if((block_size % ctx.get_current_device().get_wavefront_size()) != 0)
                        continue;
                    tc.solutions.push_back({{"tile_h", tile_h}, {"tile_w", tile_w}});
                }
            }
        }
        else
        {
            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 32}});
            tc.solutions.push_back({{"tile_h", 32}, {"tile_w", 32}});
            tc.solutions.push_back({{"tile_h", 12}, {"tile_w", 32}});
            tc.solutions.push_back({{"tile_h", 24}, {"tile_w", 16}});
            // tc.solutions.push_back({{"tile_h", 20}, {"tile_w", 8}});
            tc.solutions.push_back({{"tile_h", 32}, {"tile_w", 4}});

            // tc.solutions.push_back({{"tile_h", 16}, {"tile_w", 32}});
            // tc.solutions.push_back({{"tile_h", 64}, {"tile_w", 16}});
        }
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
