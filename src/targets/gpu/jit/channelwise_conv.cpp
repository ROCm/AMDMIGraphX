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
        channelwise_conv<index_ints<${tile}>, ${ntiles}, ${tile_c}, ${nrows}, ${cvec}>(index_ints<${tile}>{}, index_ints<${padding}>{}, ${post}, output, x, w, inputs...);
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
    // Filter taps along the first spatial dim; more than one lets row runs reuse the halo
    std::size_t row_taps() const { return num_spatial > 1 ? inputs[1].lens()[2] : 1; }

    // Channel tiles must divide the channels and stay within one input channel
    std::vector<std::size_t> channel_tiles(std::size_t max_tile) const
    {
        if(not channels_last())
            return {1};
        return divisors_up_to(multiplier() > 1 ? multiplier() : channels(), max_tile);
    }

    std::size_t default_channel_tile() const { return channel_tiles(32).back(); }

    // Filter taps per channel
    std::size_t taps() const
    {
        const auto& w_lens = inputs[1].lens();
        return std::accumulate(
            w_lens.begin() + 2, w_lens.end(), std::size_t{1}, std::multiplies<>{});
    }

    // Channels per lane that form a native (at most 16-byte) vector along a contiguous
    // channel dim and divide the channel tile
    std::vector<std::size_t> channel_vectors(std::size_t tile_c) const
    {
        std::vector<std::size_t> result = {1};
        if(not channels_last())
            return result;
        auto type_size = inputs.front().type_size();
        for(auto v : {2, 4, 8})
        {
            if(tile_c % v == 0 and v * type_size <= 16)
                result.push_back(v);
        }
        return result;
    }

    // Widest channel vector whose per-lane weights stay within a register budget
    std::size_t default_channel_vector(std::size_t tile_c) const
    {
        auto candidates = channel_vectors(tile_c);
        auto type_size  = inputs.front().type_size();
        auto it         = std::find_if(candidates.rbegin(), candidates.rend(), [&](auto v) {
            return v * taps() * type_size <= 256;
        });
        return it == candidates.rend() ? 1 : *it;
    }

    // Shared memory needed for the input halo of one block
    std::size_t halo_bytes(std::size_t tile_h,
                           std::size_t tile_w,
                           std::size_t noutputs,
                           std::size_t tile_c,
                           std::size_t nrows) const
    {
        const auto& w_lens = inputs[1].lens();
        auto tile          = spatial_tile(num_spatial, tile_h, tile_w);
        if(num_spatial > 1)
            tile.front() *= nrows;
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

// Candidate tuning solutions for a channelwise conv problem
struct channelwise_conv_solutions
{
    channelwise_conv_problem problem;
    std::size_t wave;
    std::vector<value> solutions = {};

    static constexpr std::size_t max_block = 1024;
    static constexpr std::size_t max_lds   = 64 * 1024;

    // Add a solution when its block fills whole waves and its halo fits in shared memory
    void add(std::size_t tile_h,
             std::size_t tile_w,
             std::size_t noutputs,
             std::size_t tile_c,
             std::size_t nrows,
             std::size_t cvec)
    {
        auto block_size = tile_h * tile_w * (tile_c / cvec);
        if(block_size < wave or block_size > max_block or (block_size % wave) != 0)
            return;
        if(problem.halo_bytes(tile_h, tile_w, noutputs, tile_c, nrows) > max_lds)
            return;
        solutions.push_back({{"tile_h", tile_h},
                             {"tile_w", tile_w},
                             {"noutputs", noutputs},
                             {"tile_c", tile_c},
                             {"nrows", nrows},
                             {"cvec", cvec}});
    }

    void add(std::size_t tile_h,
             std::size_t tile_w,
             std::size_t noutputs,
             std::size_t tile_c,
             std::size_t nrows)
    {
        add(tile_h, tile_w, noutputs, tile_c, nrows, problem.default_channel_vector(tile_c));
    }

    // Row runs only pay off when the filter extends along the rows
    std::vector<std::size_t> row_runs() const
    {
        if(problem.row_taps() > 1)
            return {1, 2, 4, 8};
        return {1};
    }

    // Every outputs-per-lane, row-run and channel-vector variant of a lane tile
    void add_variants(std::size_t tile_h, std::size_t tile_w, std::size_t tile_c)
    {
        for(auto opt : {1, 2, 4, 8})
        {
            for(auto nrows : row_runs())
            {
                for(auto cvec : problem.channel_vectors(tile_c))
                    add(tile_h, tile_w, opt, tile_c, nrows, cvec);
            }
        }
    }

    void add_exhaustive()
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
                    add_variants(tile_h, tile_w, tile_c);
            }
        }
    }

    void add_channels_last()
    {
        // Lanes run over channels first, so blocks need fewer spatial lanes and
        // more outputs per lane to amortize the per-lane setup
        auto tile_c = problem.default_channel_tile();
        auto lanes  = tile_c / problem.default_channel_vector(tile_c);
        for(auto tile_h : {2, 4, 8, 16})
        {
            for(auto tile_w : {16, 32, 64})
            {
                if(tile_h * tile_w * lanes < 128)
                    continue;
                for(auto opt : {4, 8, 16})
                    add(tile_h, tile_w, opt, tile_c, 1);
            }
        }
        // Strided single-channel tiles still win when the data stays in cache and the
        // filter is large enough for per-lane weights to dominate
        add(8, 32, 1, 1, 1);
        add(16, 16, 16, 1, 1);
        add(32, 16, 8, 1, 1);
        if(problem.row_taps() > 1)
        {
            add(4, 32, 4, tile_c, 2);
            add(4, 32, 4, tile_c, 4);
            add(2, 32, 4, tile_c, 4);
            add(4, 16, 8, tile_c, 4);
            add(8, 32, 1, 1, 2);
            add(8, 32, 1, 1, 4);
        }
    }

    void add_channels_first()
    {
        solutions.push_back({{"tile_h", 8}, {"tile_w", 32}, {"noutputs", 1}});

        solutions.push_back({{"tile_h", 8}, {"tile_w", 8}, {"noutputs", 8}});
        solutions.push_back({{"tile_h", 8}, {"tile_w", 16}, {"noutputs", 2}});
        solutions.push_back({{"tile_h", 8}, {"tile_w", 64}, {"noutputs", 4}});
        solutions.push_back({{"tile_h", 8}, {"tile_w", 64}, {"noutputs", 8}});
        solutions.push_back({{"tile_h", 16}, {"tile_w", 8}, {"noutputs", 4}});
        solutions.push_back({{"tile_h", 16}, {"tile_w", 16}, {"noutputs", 2}});
        solutions.push_back({{"tile_h", 16}, {"tile_w", 64}, {"noutputs", 4}});
        solutions.push_back({{"tile_h", 32}, {"tile_w", 16}, {"noutputs", 8}});
        solutions.push_back({{"tile_h", 32}, {"tile_w", 32}, {"noutputs", 1}});
        solutions.push_back({{"tile_h", 40}, {"tile_w", 12}, {"noutputs", 1}});
        solutions.push_back({{"tile_h", 48}, {"tile_w", 16}, {"noutputs", 1}});
        solutions.push_back({{"tile_h", 56}, {"tile_w", 4}, {"noutputs", 1}});
        solutions.push_back({{"tile_h", 76}, {"tile_w", 8}, {"noutputs", 8}});
        solutions.push_back({{"tile_h", 128}, {"tile_w", 8}, {"noutputs", 8}});
        if(problem.row_taps() > 1)
        {
            add(8, 32, 1, 1, 2);
            add(8, 32, 1, 1, 4);
            add(4, 32, 1, 1, 4);
            add(8, 16, 2, 1, 4);
            add(16, 16, 2, 1, 2);
            add(8, 64, 4, 1, 2);
        }
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
        // Consecutive output rows per lane
        auto nrows = v.get("nrows", std::size_t{1});
        // Channels per lane
        auto cvec = v.get("cvec", problem.default_channel_vector(tile_c));
        if(tile_c % cvec != 0)
            MIGRAPHX_THROW("channelwise_conv: channel vector must divide the channel tile");

        // Output tile = lane tile with the first spatial dim scaled by nrows and the
        // last one by noutputs
        std::vector<std::size_t> output_tile_sizes = tile_sizes;
        if(num_spatial > 1)
            output_tile_sizes.front() *= nrows;
        output_tile_sizes.back() *= noutputs;

        std::size_t block_size = std::accumulate(
            tile_sizes.begin(), tile_sizes.end(), tile_c / cvec, std::multiplies<>());

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
                                       {"nrows", std::to_string(nrows)},
                                       {"cvec", std::to_string(cvec)},
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
        channelwise_conv_solutions solutions{
            {shapes, op.to_value().at("num_spatial").to<std::size_t>()},
            ctx.get_current_device().get_wavefront_size()};
        if(exhaustive)
            solutions.add_exhaustive();
        else if(solutions.problem.channels_last())
            solutions.add_channels_last();
        else
            solutions.add_channels_first();
        tc.solutions = std::move(solutions.solutions);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
