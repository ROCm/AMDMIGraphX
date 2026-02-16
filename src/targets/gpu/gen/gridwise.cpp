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
#include <migraphx/stringutils.hpp>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

static std::string get_reduce_op(const std::string& name)
{
    if(name == "reduce_sum" or name == "reduce_mean")
        return "sum";
    if(name == "reduce_max")
        return "max";
    if(name == "reduce_min")
        return "min";
    if(name == "reduce_prod")
        return "product";
    return "";
}

static std::string select_reduce_algo(const shape& input_shape,
                                      std::size_t reduce_elements,
                                      std::size_t wavefront_size = 64)
{
    // Check for strided reduction
    auto strides    = input_shape.strides();
    auto lens       = input_shape.lens();
    bool is_strided = true;
    for(std::size_t i = 0; i < lens.size(); ++i)
    {
        if(lens[i] > 1 and strides[i] <= 2)
        {
            is_strided = false;
            break;
        }
    }
    if(is_strided and reduce_elements > 1)
        return "lane";
    if(reduce_elements <= wavefront_size)
        return "wave";
    return "block";
}

void gen_gridwise::apply(module& m) const
{
    // Gridwise pass: first lowering stage from MIGraphX IR to gen IR.
    // - Adds an output buffer parameter (high-level IR doesn't have output buffers)
    // - Adds a gpu::gen::copy from the computation result to the output buffer
    // - Detects reduce ops, selects algorithm, replaces with gridwise_reduce
    // - Inserts workgroup_id for multi-tile distribution

    // Find the return instruction
    auto ret =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "@return"; });
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

    // Detect and replace reduce operations with gridwise_reduce
    std::vector<instruction_ref> reduce_ops;
    for(auto ins : iterator_for(m))
    {
        auto rop = get_reduce_op(ins->name());
        if(not rop.empty())
            reduce_ops.push_back(ins);
    }

    for(auto ins : reduce_ops)
    {
        auto input_shape    = ins->inputs().front()->get_shape();
        auto output_shape   = ins->get_shape();
        auto input_elements = input_shape.elements();
        auto output_elements = output_shape.elements();
        auto reduce_elements = input_elements / output_elements;

        auto rop  = get_reduce_op(ins->name());
        auto algo = select_reduce_algo(input_shape, reduce_elements);

        std::size_t block_size = 256;
        if(algo == "wave")
            block_size = 64;

        auto gw_reduce = m.insert_instruction(
            ins,
            make_op("gpu::gen::gridwise_reduce",
                    {{"op", rop},
                     {"algo", algo},
                     {"reduce_elements", reduce_elements},
                     {"block_size", block_size}}),
            ins->inputs());
        m.replace_instruction(ins, gw_reduce);
    }

    // Collect parameter shapes for tile config (for pointwise tiling)
    auto param_shapes = m.get_parameter_shapes();
    std::vector<shape> shapes;
    for(const auto& p : param_shapes)
        shapes.push_back(p.second);

    auto config = compute_tile_config(shapes);

    // For multi-tile distribution, insert workgroup_id (only for non-reduce pointwise)
    if(config.ntiles > 0 and reduce_ops.empty())
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
