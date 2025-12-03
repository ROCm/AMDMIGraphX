/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/array.hpp>
#include <migraphx/ranges.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

static std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

static std::size_t compute_tile_factor(std::size_t r, std::size_t max_size = 64)
{
    std::size_t n = 1;
    auto factors  = make_array<std::size_t>(2, 3, 5, 7, 11);
    while(n < max_size)
    {
        auto it = std::find_if(factors.begin(), factors.end(), [&](auto d) { return r % d == 0; });
        if(it == factors.end())
            break;
        r /= *it;
        n *= *it;
    }
    return n;
}

tile_config tile_config::compute(const std::vector<shape>& inputs, std::size_t /*noutputs*/)
{
    tile_config result;

    if(inputs.empty())
        return result;

    auto ndim = inputs.front().ndim();

    // Find fast axis for each input
    std::vector<std::size_t> faxes;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(faxes), [](const shape& s) {
        return migraphx::gpu::gen::find_fast_axis(s);
    });

    result.axis = std::accumulate(
        faxes.begin(), faxes.end(), ndim, [](auto a, auto b) { return std::min(a, b); });

    if(result.axis >= (ndim - 1))
        return result;

    const auto& s = inputs.front();
    auto dim1     = compute_tile_factor(s.lens()[result.axis]);
    auto dim2     = compute_tile_factor(s.lens().back(), 4096 / dim1);

    if(dim1 == 1 or dim2 == 1)
        return result;

    result.inner = s.lens();
    std::fill(result.inner.begin(), result.inner.end(), 1);
    result.inner[result.axis] = dim1;
    result.inner.back()       = dim2;

    result.outer = s.lens();
    result.outer[result.axis] /= dim1;
    result.outer.back() /= dim2;

    auto tile_size = dim1 * dim2;
    result.ntiles  = s.elements() / tile_size;

    // Equivalent to dim1 * (dim2 + 1) to avoid bank conflicts
    auto tile_bytes = (tile_size + dim1) * s.type_size();
    if(tile_bytes > 65536)
        return tile_config{};

    result.block_size = std::min<std::size_t>(256, integer_divide_ceil(tile_size / 4, 64) * 64);
    result.tile_dims  = {dim1, dim2};

    return result;
}

// Reduction operations that need special handling
static const std::unordered_set<std::string>& reduction_ops()
{
    static const std::unordered_set<std::string> names = {
        "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod", "reduce_any", "reduce_all"};
    return names;
}

// Check if an instruction is a reduction
static bool is_reduction(instruction_ref ins)
{
    return contains(reduction_ops(), ins->name());
}

// Check if module contains any reduction operations
static bool contains_reduction(const module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(is_reduction(ins))
            return true;
    }
    return false;
}

// Compute reduction configuration
struct reduce_config
{
    std::vector<std::size_t> reduce_axes;
    std::size_t reduce_elements = 0;
    std::size_t output_elements = 0;
    std::string algo            = "lane"; // lane, wave, block
    std::size_t block_size      = 64;

    bool is_valid() const { return not reduce_axes.empty(); }
};

static reduce_config compute_reduce_config(const module& m, std::size_t wavefront_size = 64)
{
    reduce_config result;

    // Find the reduction instruction
    instruction_ref reduce_ins;
    for(auto ins : iterator_for(m))
    {
        if(is_reduction(ins))
        {
            reduce_ins = ins;
            break;
        }
    }

    if(reduce_ins == m.end())
        return result;

    // Get axes from the reduction
    auto v    = reduce_ins->get_operator().to_value();
    auto axes = v.at("axes").to_vector<std::int64_t>();

    // Get input shape
    auto input_shape = reduce_ins->inputs().front()->get_shape();
    auto ndim        = input_shape.ndim();

    // Normalize negative axes
    for(auto& axis : axes)
    {
        if(axis < 0)
            axis += ndim;
    }
    std::sort(axes.begin(), axes.end());

    // Compute reduce elements
    std::size_t reduce_elements = 1;
    for(auto axis : axes)
    {
        reduce_elements *= input_shape.lens()[axis];
    }

    result.reduce_axes    = std::vector<std::size_t>(axes.begin(), axes.end());
    result.reduce_elements = reduce_elements;
    result.output_elements = input_shape.elements() / reduce_elements;

    // Select algorithm based on reduction size
    if(reduce_elements <= wavefront_size)
    {
        result.algo       = "wave";
        result.block_size = wavefront_size;
    }
    else if(reduce_elements <= 256)
    {
        result.algo       = "block";
        result.block_size = std::min<std::size_t>(256, reduce_elements);
    }
    else
    {
        result.algo       = "lane";
        result.block_size = 256;
    }

    return result;
}

void gen_tiling::apply(module& m) const
{
    // First check if module contains reductions - handle differently
    if(contains_reduction(m))
    {
        // For reductions, we don't tile the same way as copies
        // The reduction lowering pass will handle the tiling pattern
        // Just compute and store the reduction config for later use
        auto config = compute_reduce_config(m);
        (void)config; // Will be used by lower pass
        return;
    }

    // Collect copy instructions to tile
    std::vector<instruction_ref> copies_to_tile;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::copy")
        {
            copies_to_tile.push_back(ins);
        }
    }

    for(auto ins : copies_to_tile)
    {
        auto inputs = ins->inputs();
        if(inputs.size() != 2)
            continue;

        auto src       = inputs[0];
        auto dst       = inputs[1];
        auto src_shape = src->get_shape();

        // Skip if already tiled (inputs are tile_region)
        if(src->name() == "gpu::gen::tile_region")
            continue;

        // Only tile multi-dimensional tensors
        if(src_shape.ndim() <= 1)
            continue;

        // Compute tiling configuration
        std::vector<shape> shapes = {src_shape, dst->get_shape()};
        auto config               = tile_config::compute(shapes, 1);

        if(not config.is_tiled())
            continue;

        // Insert workgroup_id
        auto wg_id = m.insert_instruction(ins, make_op("gpu::gen::workgroup_id"));

        // Create tile_region for source
        auto src_tile =
            m.insert_instruction(ins,
                                 make_op("gpu::gen::tile_region",
                                         {{"tile_dims", config.tile_dims}, {"axis", config.axis}}),
                                 src,
                                 wg_id);

        // Create tile_region for destination
        auto dst_tile =
            m.insert_instruction(ins,
                                 make_op("gpu::gen::tile_region",
                                         {{"tile_dims", config.tile_dims}, {"axis", config.axis}}),
                                 dst,
                                 wg_id);

        // Replace the copy's inputs with tile_regions
        m.replace_instruction(ins, ins->get_operator(), {src_tile, dst_tile});
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
