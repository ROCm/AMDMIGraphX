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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const channelwise_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/channelwise_conv.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto output, auto x, auto w, auto... inputs) {
        channelwise_conv<index_ints<${tile}>, ${ntiles}, ${tile_c}>(index_ints<${tile}>{}, index_ints<${padding}>{}, ${post}, output, x, w, inputs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

// Divisors of `n` up to `max_value`, ascending
static std::vector<std::size_t> divisors_up_to(std::size_t n, std::size_t max_value)
{
    std::vector<std::size_t> candidates(std::min(n, max_value));
    std::iota(candidates.begin(), candidates.end(), std::size_t{1});
    std::vector<std::size_t> result;
    std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(result), [&](auto d) {
        return n % d == 0;
    });
    return result;
}

// Per-lane spatial tile: tile_h on the first spatial dim, tile_w on the last
static std::vector<std::size_t>
spatial_tile(std::size_t num_spatial, std::size_t tile_h, std::size_t tile_w)
{
    std::vector<std::size_t> result(num_spatial, 1);
    if(num_spatial > 1)
        result.front() = tile_h;
    result.back() = tile_w;
    return result;
}

struct channelwise_conv_problem
{
    std::vector<shape> inputs;
    std::size_t num_spatial = 2;

    std::size_t channels() const { return inputs.back().lens()[1]; }
    // Output channels per input channel
    std::size_t multiplier() const { return channels() / inputs.front().lens()[1]; }
    // Channel is the fastest dim in memory, so tile over channels to coalesce accesses
    bool channels_last() const { return find_permutation(inputs.back()).back() == 1; }

    // Channel tiles must divide the channels and stay within one input channel
    std::vector<std::size_t> channel_tiles(std::size_t max_tile) const
    {
        if(not channels_last())
            return {1};
        return divisors_up_to(multiplier() > 1 ? multiplier() : channels(), max_tile);
    }

    std::size_t default_channel_tile() const { return channel_tiles(32).back(); }

    // Shared memory needed for the input halo of one block
    std::size_t halo_bytes(std::size_t tile_h,
                           std::size_t tile_w,
                           std::size_t noutputs,
                           std::size_t tile_c) const
    {
        const auto& w_lens = inputs[1].lens();
        auto tile          = spatial_tile(num_spatial, tile_h, tile_w);
        tile.back() *= noutputs;
        auto halo = std::inner_product(tile.begin(),
                                       tile.end(),
                                       w_lens.begin() + 2,
                                       std::size_t{1},
                                       std::multiplies<>{},
                                       [](auto t, auto k) { return t + k - 1; });
        return halo * (multiplier() > 1 ? 1 : tile_c) * inputs.front().type_size();
    }
};

struct channelwise_conv_compiler : compiler<channelwise_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::channelwise_conv", "channelwise_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto num_spatial       = v.at("num_spatial").to<std::size_t>();
        const auto& out_s      = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = v.get("kernel", std::string{"channelwise_conv_kernel"});
        options.virtual_inputs = inputs;

        const auto& out_lens = out_s.lens();
        channelwise_conv_problem problem{inputs, num_spatial};

        // Thread block tile dimensions
        auto tile_sizes = spatial_tile(num_spatial,
                                       v.get("tile_h", std::size_t{8}),
                                       v.get("tile_w", num_spatial == 1 ? 256 : 32));

        // Outputs per lane along W (last spatial dim)
        auto noutputs = v.get("noutputs", std::size_t{4});
        // Channels per block
        auto tile_c = v.get("tile_c", problem.default_channel_tile());

        // Output tile = lane tile with last dim scaled by noutputs
        std::vector<std::size_t> output_tile_sizes = tile_sizes;
        output_tile_sizes.back() *= noutputs;

        std::size_t block_size =
            std::accumulate(tile_sizes.begin(), tile_sizes.end(), tile_c, std::multiplies<>());

        // Blocks: N * (C_out / tile_c) * prod(ceil(out_spatial / output_tile))
        auto num_blocks = std::inner_product(
            out_lens.begin() + 2,
            out_lens.end(),
            output_tile_sizes.begin(),
            out_lens[0] * (out_lens[1] / tile_c),
            std::multiplies<>{},
            [](auto out_spatial, auto tile) { return (out_spatial + tile - 1) / tile; });

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto padding = v.get("padding", std::vector<std::size_t>{});
        if(padding.size() < 2 * num_spatial)
            padding.resize(2 * num_spatial, 0);

        auto src = interpolate_string(channelwise_conv_kernel,
                                      {{"tile", to_string_range(tile_sizes)},
                                       {"ntiles", std::to_string(noutputs)},
                                       {"tile_c", std::to_string(tile_c)},
                                       {"padding", to_string_range(padding)},
                                       {"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"preamble", v.get("preamble", std::string{})}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto v = op.to_value();
        for(const auto& x : solution)
            v.insert(x);
        if(not ins->module_inputs().empty())
        {
            auto* pm      = ins->module_inputs().front();
            v["preamble"] = generate_pointwise(*pm, "post_channelwise_conv");
            v["post"]     = "MIGRAPHX_LIFT(post_channelwise_conv)";
            v["kernel"]   = "channelwise_conv_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation& op,
                                              bool exhaustive) const
    {
        tuning_config tc;
        auto shapes = to_shapes(ins->inputs());
        tc.problem  = to_value(shapes);
        channelwise_conv_problem problem{shapes, op.to_value().at("num_spatial").to<std::size_t>()};

        const std::size_t max_block = 1024;
        const std::size_t max_lds   = 64 * 1024;
        auto wave                   = ctx.get_current_device().get_wavefront_size();
        auto add_solution =
            [&](std::size_t tile_h, std::size_t tile_w, std::size_t noutputs, std::size_t tile_c) {
                auto block_size = tile_h * tile_w * tile_c;
                if(block_size < wave or block_size > max_block or (block_size % wave) != 0)
                    return;
                if(problem.halo_bytes(tile_h, tile_w, noutputs, tile_c) > max_lds)
                    return;
                tc.solutions.push_back({{"tile_h", tile_h},
                                        {"tile_w", tile_w},
                                        {"noutputs", noutputs},
                                        {"tile_c", tile_c}});
            };

        if(exhaustive)
        {
            std::vector<std::size_t> sizes;
            if(problem.channels_last())
                sizes = {1, 2};
            transform(range(1, 64), std::back_inserter(sizes), [](auto i) { return i * 4; });
            for(auto tile_c : problem.channel_tiles(64))
            {
                for(auto tile_h : sizes)
                {
                    for(auto tile_w : sizes)
                    {
                        for(auto opt : {1, 2, 4, 8})
                            add_solution(tile_h, tile_w, opt, tile_c);
                    }
                }
            }
        }
        else if(problem.channels_last())
        {
            // Lanes run over channels first, so blocks need fewer spatial lanes and
            // more outputs per lane to amortize the per-lane setup
            auto tile_c = problem.default_channel_tile();
            for(auto tile_h : {2, 4, 8, 16})
            {
                for(auto tile_w : {16, 32, 64})
                {
                    if(tile_h * tile_w * tile_c < 128)
                        continue;
                    for(auto opt : {4, 8, 16})
                        add_solution(tile_h, tile_w, opt, tile_c);
                }
            }
            // Strided single-channel tiles still win when the data stays in cache and the
            // filter is large enough for per-lane weights to dominate
            add_solution(8, 32, 1, 1);
            add_solution(16, 16, 16, 1);
            add_solution(32, 16, 8, 1);
        }
        else
        {
            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 32}, {"noutputs", 1}});

            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 8}, {"noutputs", 8}});
            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 16}, {"noutputs", 2}});
            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 64}, {"noutputs", 4}});
            tc.solutions.push_back({{"tile_h", 8}, {"tile_w", 64}, {"noutputs", 8}});
            tc.solutions.push_back({{"tile_h", 16}, {"tile_w", 8}, {"noutputs", 4}});
            tc.solutions.push_back({{"tile_h", 16}, {"tile_w", 16}, {"noutputs", 2}});
            tc.solutions.push_back({{"tile_h", 16}, {"tile_w", 64}, {"noutputs", 4}});
            tc.solutions.push_back({{"tile_h", 32}, {"tile_w", 16}, {"noutputs", 8}});
            tc.solutions.push_back({{"tile_h", 32}, {"tile_w", 32}, {"noutputs", 1}});
            tc.solutions.push_back({{"tile_h", 40}, {"tile_w", 12}, {"noutputs", 1}});
            tc.solutions.push_back({{"tile_h", 48}, {"tile_w", 16}, {"noutputs", 1}});
            tc.solutions.push_back({{"tile_h", 56}, {"tile_w", 4}, {"noutputs", 1}});
            tc.solutions.push_back({{"tile_h", 76}, {"tile_w", 8}, {"noutputs", 8}});
            tc.solutions.push_back({{"tile_h", 128}, {"tile_w", 8}, {"noutputs", 8}});
        }
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
