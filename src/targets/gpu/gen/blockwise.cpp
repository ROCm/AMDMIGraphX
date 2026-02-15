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
#include <migraphx/gpu/gen/blockwise.hpp>
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

void gen_blockwise::apply(module& m) const
{
    // Blockwise pass: manages work within a single workgroup.
    // - Inserts tile_region ops to create tiled views when tiling is active.
    // - Manages LDS allocations for data sharing between lanes.
    // - Inserts barriers for synchronization.

    auto param_shapes = m.get_parameter_shapes();
    std::vector<shape> shapes;
    for(const auto& p : param_shapes)
        shapes.push_back(p.second);

    auto config = compute_tile_config(shapes);

    if(config.ntiles == 0 or config.tile_dims.empty())
        return; // No tiling needed at blockwise level

    // Find workgroup_id instruction
    auto wg_id = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::gen::workgroup_id"; });
    if(wg_id == m.end())
        return;

    instruction_ref wg_id_ref = wg_id;

    // Insert tile_region for each parameter
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@param")
            continue;

        auto s = ins->get_shape();
        if(s.ndim() < 2)
            continue;

        // Insert tile_region after the parameter
        auto tile =
            m.insert_instruction(std::next(ins),
                                 make_op("gpu::gen::tile_region",
                                         {{"tile_dims", config.tile_dims}, {"axis", config.axis}}),
                                 ins,
                                 wg_id_ref);

        // Replace uses of the parameter with the tiled region
        // (except the tile_region itself)
        for(auto output : ins->outputs())
        {
            if(output == tile)
                continue;
            if(output->name() == "gpu::gen::tile_region")
                continue;
            instruction::replace_argument(output, ins, tile);
        }
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
