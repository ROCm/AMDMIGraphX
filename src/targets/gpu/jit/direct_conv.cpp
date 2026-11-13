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
 *
 */
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/math.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/serialize.hpp>
#include <numeric>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const direct_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/direct_conv.hpp>
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
        direct_conv<index_ints<${tile}>, ${noutputs}, ${tile_k}, ${nrows}, ${kvec}, ${tile_c}>(index_ints<${tile}>{}, index_ints<${padding}>{}, index_ints<${stride}>{}, index_ints<${dilation}>{}, ${post}, output, x, w, inputs...);
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

static bool is_power_of_2(std::size_t n) { return n > 0 and (n & (n - 1)) == 0; }

// One tuning solution: a lane computes kvec output channels of nrows rows by noutputs
// columns spaced tile_w apart; a block covers tile_k output channels over a
// (tile_h * nrows, tile_w * noutputs) spatial tile and stages tile_c input channels at a time
struct direct_conv_config
{
    std::size_t tile_h   = 1;
    std::size_t tile_w   = 64;
    std::size_t noutputs = 1;
    std::size_t nrows    = 1;
    std::size_t tile_k   = 1;
    std::size_t kvec     = 1;
    std::size_t tile_c   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tile_h, "tile_h"),
                    f(self.tile_w, "tile_w"),
                    f(self.noutputs, "noutputs"),
                    f(self.nrows, "nrows"),
                    f(self.tile_k, "tile_k"),
                    f(self.kvec, "kvec"),
                    f(self.tile_c, "tile_c"));
    }

    std::size_t block_size() const { return tile_h * tile_w * (tile_k / kvec); }
    std::size_t accumulators() const { return kvec * nrows * noutputs; }
};

struct direct_conv_problem
{
    std::vector<shape> inputs;
    std::size_t num_spatial = 2;
    std::vector<std::size_t> stride;
    std::vector<std::size_t> dilation;
    // Begin padding per spatial dim followed by end padding
    std::vector<std::size_t> padding;
    std::size_t max_lds_bytes = 0;

    static direct_conv_problem
    from_value(const std::vector<shape>& inputs, const value& v, std::size_t max_lds_bytes)
    {
        auto num_spatial = inputs.front().ndim() - 2;
        auto stride      = v.at("stride").to_vector<std::size_t>();
        auto dilation    = v.at("dilation").to_vector<std::size_t>();
        auto padding     = v.at("padding").to_vector<std::size_t>();
        if(stride.size() != num_spatial or dilation.size() != num_spatial)
            MIGRAPHX_THROW("direct_conv: stride and dilation need one value per spatial dim");
        if(padding.size() == num_spatial)
        {
            auto begin = padding;
            padding.insert(padding.end(), begin.begin(), begin.end());
        }
        if(padding.size() != 2 * num_spatial)
            MIGRAPHX_THROW("direct_conv: padding needs one or two values per spatial dim");
        return {inputs,
                num_spatial,
                std::move(stride),
                std::move(dilation),
                std::move(padding),
                max_lds_bytes};
    }

    // Tuning cache key: stride, dilation and padding change the halo and indexing
    value key() const
    {
        return {{"inputs", to_value(inputs)},
                {"padding", padding},
                {"stride", stride},
                {"dilation", dilation}};
    }

    std::size_t in_channels() const { return inputs.front().lens()[1]; }
    std::size_t out_channels() const { return inputs.back().lens()[1]; }
    std::size_t type_size() const { return inputs.front().type_size(); }
    const std::vector<std::size_t>& weight_lens() const { return inputs[1].lens(); }

    // Input channels the input channel tile should be a multiple of: the kernel reduces
    // half channel pairs with one dot2 instruction when the tile is even
    std::size_t channel_step() const
    {
        if(inputs.front().type() == shape::half_type and in_channels() % 2 == 0)
            return 2;
        return 1;
    }

    // Filter taps per input channel
    std::size_t taps() const
    {
        return std::accumulate(
            weight_lens().begin() + 2, weight_lens().end(), std::size_t{1}, std::multiplies<>{});
    }

    // Input extent one output reads along spatial dim i: the dilated filter
    std::size_t reach(std::size_t i) const { return (weight_lens()[i + 2] - 1) * dilation[i] + 1; }

    // Filter taps along the first spatial dim; more than one lets row runs reuse the halo
    std::size_t row_taps() const { return num_spatial > 1 ? weight_lens()[2] : 1; }

    // Halo rows a lane keeps in registers per filter column and output column: those its
    // row run's taps read
    std::size_t halo_rows(std::size_t nrows) const
    {
        auto span = (nrows - 1) * stride.front() + (row_taps() - 1) * dilation.front() + 1;
        return std::min(span, nrows * row_taps());
    }

    // Lanes of a block along the spatial dims: tile_h on the first, tile_w on the last
    std::vector<std::size_t> lane_tile(const direct_conv_config& c) const
    {
        std::vector<std::size_t> result(num_spatial, 1);
        if(num_spatial > 1)
            result.front() = c.tile_h;
        result.back() = c.tile_w;
        return result;
    }

    // Output tile of a block: the lane tile with the rows and columns per lane
    std::vector<std::size_t> output_tile(const direct_conv_config& c) const
    {
        auto result = lane_tile(c);
        if(num_spatial > 1)
            result.front() *= c.nrows;
        result.back() *= c.noutputs;
        return result;
    }

    // Shared memory for the halo of tile_c input channels plus their weights. The buffers
    // align to 16 bytes, or to a lane's weight vector when it is a wider power of two.
    std::size_t lds_bytes(const direct_conv_config& c) const
    {
        auto tile = output_tile(c);
        auto dims = range(num_spatial);
        auto halo = transform_accumulate(
            dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>{}, [&](auto i) {
                return (tile[i] - 1) * stride[i] + reach(i);
            });
        std::size_t dot_size =
            inputs.front().type() == shape::half_type and c.tile_c % 2 == 0 ? 2 : 1;
        auto w_vec        = c.kvec * dot_size * type_size();
        std::size_t align = w_vec > 16 and is_power_of_2(w_vec) ? w_vec : 16;
        return ceil_mul_of(halo * c.tile_c * type_size(), align) +
               ceil_mul_of(c.tile_k * taps() * c.tile_c * type_size(), align);
    }

    std::size_t num_blocks(const direct_conv_config& c) const
    {
        const auto& out_lens = inputs.back().lens();
        auto tile            = output_tile(c);
        return std::inner_product(out_lens.begin() + 2,
                                  out_lens.end(),
                                  tile.begin(),
                                  out_lens[0] * integer_divide_ceil(out_lens[1], c.tile_k),
                                  std::multiplies<>{},
                                  integer_divide_ceil);
    }

    // Whether the kernel can run the configuration and keeps its registers in budget
    bool valid(const direct_conv_config& c, std::size_t wave) const
    {
        // A solution read from a problem cache may hold zeros
        if(c.kvec == 0 or c.tile_c == 0)
            return false;
        if(num_spatial < 2 and (c.tile_h != 1 or c.nrows != 1))
            return false;
        if(c.tile_k % c.kvec != 0)
            return false;
        if(in_channels() % c.tile_c != 0)
            return false;
        // Power-of-two channel vectors load as one vector, which shared memory only aligns
        // up to 16 bytes; other widths load element by element
        if(c.kvec > 8 or (is_power_of_2(c.kvec) and c.kvec * type_size() > 16))
            return false;
        if(c.accumulators() > 64 or c.noutputs * halo_rows(c.nrows) > 64)
            return false;
        auto block = c.block_size();
        if(block < wave or block > 1024 or block % wave != 0)
            return false;
        return lds_bytes(c) <= max_lds_bytes;
    }
};

struct direct_conv_solutions
{
    direct_conv_problem problem;
    std::size_t wave;
    std::vector<value> solutions = {};
    std::set<value> seen         = {};

    void add(direct_conv_config config)
    {
        if(problem.num_spatial < 2)
        {
            config.tile_w *= config.tile_h;
            config.tile_h = 1;
            config.nrows  = 1;
        }
        if(not problem.valid(config, wave))
            return;
        auto v = to_value(config);
        if(seen.insert(v).second)
            solutions.push_back(std::move(v));
    }

    // Largest divisor of the input channels up to max_tile that is a multiple of the channel
    // step and whose shared memory fits, else 1
    std::size_t fit_tile_c(direct_conv_config c, std::size_t max_tile) const
    {
        auto tiles = divisors_up_to(problem.in_channels(), max_tile);
        auto it    = std::find_if(tiles.rbegin(), tiles.rend(), [&](auto tile_c) {
            c.tile_c = tile_c;
            return tile_c % problem.channel_step() == 0 and
                   problem.lds_bytes(c) <= problem.max_lds_bytes;
        });
        return it == tiles.rend() ? 1 : *it;
    }

    // Output channels per lane for blocks of tile_k channels: short vectors that load
    // natively
    std::vector<std::size_t> kvecs(std::size_t tile_k) const
    {
        const std::vector<std::size_t> candidates = {4, 8};
        std::vector<std::size_t> result;
        std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(result), [&](auto v) {
            return tile_k % v == 0 and v * problem.type_size() <= 16;
        });
        if(result.empty())
            result.push_back(divisors_up_to(tile_k, 4).back());
        return result;
    }

    // Add the configuration with the largest input channel tile up to each of max_cs that
    // fits
    void add_channel_tiles(direct_conv_config c, const std::vector<std::size_t>& max_cs)
    {
        for(auto max_c : max_cs)
        {
            c.tile_c = fit_tile_c(c, max_c);
            add(c);
        }
    }

    // Few output channels: lanes compute all of them, or a native vector of them when that
    // is narrower, over wide spatial blocks
    void add_few_outputs()
    {
        auto k    = problem.out_channels();
        auto kvec = k;
        if(is_power_of_2(k) and k * problem.type_size() > 16)
            kvec = 16 / problem.type_size();
        std::vector<std::size_t> row_runs = {1, 2};
        if(problem.row_taps() > 1)
            row_runs.push_back(4);
        for(auto [tile_h, tile_w] : {std::pair<std::size_t, std::size_t>{8, 16}, {4, 32}, {8, 32}})
        {
            for(auto nrows : row_runs)
                add_channel_tiles({tile_h, tile_w, 1, nrows, k, kvec}, {4, 8, 16});
        }
    }

    // Blocks of 16 or 32 output channels (8 or 16 below 16) over narrow spatial tiles;
    // multi-row filters reuse long row runs, 1-row filters take two columns per lane
    void add_many_outputs()
    {
        std::vector<std::size_t> tile_ks = {16, 32};
        if(problem.out_channels() < 16)
            tile_ks = {8, 16};
        std::vector<std::size_t> row_runs = {4, 8};
        std::size_t noutputs              = 1;
        if(problem.row_taps() == 1)
        {
            row_runs = {1, 2};
            noutputs = 2;
        }
        for(auto tile_k : tile_ks)
        {
            for(auto kvec : kvecs(tile_k))
            {
                for(auto [tile_h, tile_w] :
                    {std::pair<std::size_t, std::size_t>{1, 32}, {2, 16}, {2, 32}})
                {
                    for(auto nrows : row_runs)
                        add_channel_tiles({tile_h, tile_w, noutputs, nrows, tile_k, kvec}, {4, 8});
                }
            }
        }
    }

    void add_default()
    {
        if(problem.out_channels() <= 8)
            add_few_outputs();
        else
            add_many_outputs();
        add_fallback();
    }

    // A small tile that fits whenever one input channel of an 8x64 output tile does,
    // so every convolution prefuse_ops offloads has a solution
    void add_fallback() { add({8, 32, 2, 1, 1, 1, 1}); }

    // Every row run, outputs per lane and input channel tile of a lane tile
    void add_variants(std::size_t tile_h, std::size_t tile_w, std::size_t tile_k, std::size_t kvec)
    {
        for(std::size_t nrows : {1, 2, 4, 8})
        {
            for(std::size_t noutputs : {1, 2, 4, 8})
                add_channel_tiles({tile_h, tile_w, noutputs, nrows, tile_k, kvec},
                                  {1, 4, 8, 16, 32});
        }
    }

    void add_exhaustive()
    {
        auto tile_ks = divisors_up_to(problem.out_channels(), 64);
        for(std::size_t t : {8, 16, 32, 64})
        {
            if(not contains(tile_ks, t))
                tile_ks.push_back(t);
        }
        for(auto tile_k : tile_ks)
        {
            for(auto kvec : divisors_up_to(tile_k, 8))
            {
                for(std::size_t tile_h : {1, 2, 4, 8, 16})
                {
                    for(std::size_t tile_w : {8, 16, 32, 64})
                        add_variants(tile_h, tile_w, tile_k, kvec);
                }
            }
        }
        add_fallback();
    }
};

struct direct_conv_compiler : compiler<direct_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::direct_conv", "direct_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto problem = direct_conv_problem::from_value(
            inputs, v, ctx.get_current_device().get_max_shared_memory_per_block());
        auto wave = ctx.get_current_device().get_wavefront_size();

        direct_conv_config config;
        if(v.contains("tile_k"))
        {
            config = from_value<direct_conv_config>(v);
        }
        else
        {
            direct_conv_solutions solutions{problem, wave};
            solutions.add_default();
            if(solutions.solutions.empty())
                MIGRAPHX_THROW("direct_conv: no valid configuration");
            config = from_value<direct_conv_config>(solutions.solutions.front());
        }
        if(not problem.valid(config, wave))
            MIGRAPHX_THROW("direct_conv: invalid configuration " + to_string(to_value(config)));

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = v.get("kernel", std::string{"direct_conv_kernel"});
        options.virtual_inputs = inputs;
        options.set_launch_params(
            v, problem.num_blocks(config) * config.block_size(), config.block_size());

        auto src = interpolate_string(direct_conv_kernel,
                                      {{"tile", to_string_range(problem.lane_tile(config))},
                                       {"noutputs", std::to_string(config.noutputs)},
                                       {"tile_k", std::to_string(config.tile_k)},
                                       {"nrows", std::to_string(config.nrows)},
                                       {"kvec", std::to_string(config.kvec)},
                                       {"tile_c", std::to_string(config.tile_c)},
                                       {"padding", to_string_range(problem.padding)},
                                       {"stride", to_string_range(problem.stride)},
                                       {"dilation", to_string_range(problem.dilation)},
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
            v["preamble"] = generate_pointwise(*pm, "post_direct_conv");
            v["post"]     = "MIGRAPHX_LIFT(post_direct_conv)";
            v["kernel"]   = "direct_conv_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation& op,
                                              bool exhaustive) const
    {
        direct_conv_solutions solutions{
            direct_conv_problem::from_value(
                to_shapes(ins->inputs()),
                op.to_value(),
                ctx.get_current_device().get_max_shared_memory_per_block()),
            ctx.get_current_device().get_wavefront_size()};
        tuning_config tc;
        tc.problem = solutions.problem.key();
        if(exhaustive)
            solutions.add_exhaustive();
        else
            solutions.add_default();
        tc.solutions = std::move(solutions.solutions);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
