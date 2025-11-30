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

void gen_tiling::apply(module& m) const
{
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
