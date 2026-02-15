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
#include <migraphx/gpu/gen/gridwise.hpp>
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

void gen_gridwise::apply(module& m) const
{
    // Gridwise pass: first lowering stage from MIGraphX IR to gen IR.
    // - Adds an output buffer parameter (high-level IR doesn't have output buffers)
    // - Adds a gpu::gen::copy from the computation result to the output buffer
    // - Inserts workgroup_id for multi-tile distribution

    // Find the return instruction
    auto ret = std::find_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "@return"; });
    if(ret == m.end())
        return;

    instruction_ref ret_ref = ret;

    // Get the return value (the result of computation)
    auto ret_input = ret_ref->inputs().front();
    auto out_shape = ret_input->get_shape();

    // Add output buffer parameter
    auto z_output = m.add_parameter("z_output", out_shape);

    // Replace @return(computation) with:
    //   copy = gpu::gen::copy(computation, z_output)
    //   @return(copy)
    auto copy = m.insert_instruction(
        ret_ref, make_op("gpu::gen::copy", {{"schedule", "per_lane"}}), ret_input, z_output);
    m.replace_instruction(ret_ref, make_op("@return"), {copy});

    // Collect parameter shapes for tile config
    auto param_shapes = m.get_parameter_shapes();
    std::vector<shape> shapes;
    for(const auto& p : param_shapes)
        shapes.push_back(p.second);

    auto config = compute_tile_config(shapes);

    // For multi-tile distribution, insert workgroup_id
    if(config.ntiles > 0)
    {
        auto first_non_param = std::find_if(
            m.begin(), m.end(), [](const auto& ins) { return ins.name() != "@param"; });
        m.insert_instruction(first_non_param, make_op("gpu::gen::workgroup_id"));
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
