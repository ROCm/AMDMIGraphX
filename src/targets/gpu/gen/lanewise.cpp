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
#include <migraphx/gpu/gen/lanewise.hpp>
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

void gen_lanewise::apply(module& m) const
{
    // Lanewise pass: lowers to per-element load/store operations.
    //
    // Input (from gridwise):
    //   x0, x1 = @param (tensor inputs)
    //   z_output = @param (output buffer)
    //   result = add(x0, x1)          (pointwise on tensors)
    //   copy = gpu::gen::copy(result, z_output)
    //   @return(copy)
    //
    // Output:
    //   x0, x1 = @param (tensor inputs)
    //   z_output = @param (output buffer)
    //   tid = gpu::gen::global_id
    //   x0_val = gpu::gen::load(x0, tid)    (per-element load)
    //   x1_val = gpu::gen::load(x1, tid)    (per-element load)
    //   result = add(x0_val, x1_val)        (scalar pointwise)
    //   store = gpu::gen::store(z_output, tid, result)
    //   @return(store)

    auto ret =
        std::find_if(m.begin(), m.end(), [](const auto& ins) { return ins.name() == "@return"; });
    if(ret == m.end())
        return;

    // Insert thread ID
    bool has_tiles = std::any_of(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::gen::tile_region"; });

    auto first_non_param = std::find_if(m.begin(), m.end(), [](const auto& ins) {
        return ins.name() != "@param" and ins.name() != "gpu::gen::workgroup_id" and
               ins.name() != "gpu::gen::tile_region";
    });

    instruction_ref tid;
    if(has_tiles)
        tid = m.insert_instruction(first_non_param, make_op("gpu::gen::local_id"));
    else
        tid = m.insert_instruction(first_non_param, make_op("gpu::gen::global_id"));

    // Step 1: Lower copy(src, dst) -> store(dst, tid, src)
    // Do this FIRST before we change pointwise op shapes.
    std::vector<instruction_ref> copies;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::copy")
            copies.push_back(ins);
    }

    for(auto copy_ins : copies)
    {
        auto src = copy_ins->inputs()[0];
        auto dst = copy_ins->inputs()[1];

        auto store = m.insert_instruction(copy_ins, make_op("gpu::gen::store"), dst, tid, src);
        m.replace_instruction(copy_ins, store);
    }

    // Step 2: Lower index transform ops (pad, gather, reverse) to gen IR.
    // These produce index transforms + loads that yield scalar values.
    std::vector<instruction_ref> index_ops;
    for(auto ins : iterator_for(m))
    {
        if(contains({"pad", "gather", "reverse"}, ins->name()))
            index_ops.push_back(ins);
    }

    std::unordered_map<instruction_ref, instruction_ref> index_remap;
    for(auto ins : index_ops)
    {
        if(ins->name() == "pad")
        {
            auto v          = ins->get_operator().to_value();
            auto pads       = v.at("pads").to_vector<std::int64_t>();
            auto input      = ins->inputs().front();
            auto input_s    = input->get_shape();
            auto pad_value  = 0.0f;
            if(v.contains("value"))
                pad_value = v.at("value").to<float>();

            // pad_index(tid) -> int64 index (-1 if out of bounds)
            auto pad_idx = m.insert_instruction(
                ins,
                make_op("gpu::gen::pad_index",
                        {{"input_shape", to_value(input_s)}, {"pads", pads}}),
                tid);
            // conditional_load(input, pad_idx, fill_value)
            auto fill_lit = m.add_literal(migraphx::literal{shape{input_s.type()}, {pad_value}});
            auto loaded   = m.insert_instruction(
                ins, make_op("gpu::gen::conditional_load"), input, pad_idx, fill_lit);
            index_remap[ins] = loaded;
        }
        else if(ins->name() == "gather")
        {
            auto v       = ins->get_operator().to_value();
            auto axis    = v.at("axis").to<std::int64_t>();
            auto data    = ins->inputs()[0];
            auto indices = ins->inputs()[1];
            auto input_s = data->get_shape();

            // gather_index(indices, tid) -> index into data
            auto g_idx = m.insert_instruction(
                ins,
                make_op("gpu::gen::gather_index",
                        {{"input_shape", to_value(input_s)}, {"axis", axis}}),
                indices,
                tid);
            // load(data, gather_idx)
            auto loaded = m.insert_instruction(
                ins, make_op("gpu::gen::load"), data, g_idx);
            index_remap[ins] = loaded;
        }
        else if(ins->name() == "reverse")
        {
            auto v       = ins->get_operator().to_value();
            auto axes    = v.at("axes").to_vector<std::int64_t>();
            auto input   = ins->inputs().front();
            auto input_s = input->get_shape();

            // reverse_index(tid) -> reversed index
            auto r_idx = m.insert_instruction(
                ins,
                make_op("gpu::gen::reverse_index",
                        {{"input_shape", to_value(input_s)}, {"axes", axes}}),
                tid);
            // load(input, reversed_idx)
            auto loaded = m.insert_instruction(
                ins, make_op("gpu::gen::load"), input, r_idx);
            index_remap[ins] = loaded;
        }
    }

    // Step 3: Insert loads for tensor-producing instructions that feed into
    // pointwise ops. This covers @param inputs and tile_region outputs.
    // Skip z_output since it's the output buffer.
    std::unordered_map<instruction_ref, instruction_ref> load_map;
    for(auto ins : iterator_for(m))
    {
        // Skip the output param
        if(ins->name() == "@param")
        {
            auto pname = ins->get_operator().to_value().at("parameter").to<std::string>();
            if(pname == "z_output")
                continue;
        }

        // Only insert loads for instructions whose outputs feed into pointwise ops
        bool needs_load = false;
        for(auto out : ins->outputs())
        {
            auto n = out->name();
            if(starts_with(n, "@") or starts_with(n, "gpu::gen::"))
                continue;
            // This output is a pointwise op consuming our tensor
            needs_load = true;
            break;
        }
        if(not needs_load)
            continue;

        // Only load from @param or gen IR instructions that produce tensors
        if(ins->name() != "@param" and not starts_with(ins->name(), "gpu::gen::"))
            continue;

        auto load     = m.insert_instruction(std::next(tid), make_op("gpu::gen::load"), ins, tid);
        load_map[ins] = load;
    }

    // Collect pointwise ops, then rebuild as new scalar instructions.
    // We can't modify in-place because intermediate shape mismatches
    // trigger validation errors.
    std::vector<instruction_ref> pw_ops;
    for(auto ins : iterator_for(m))
    {
        auto n = ins->name();
        if(starts_with(n, "@") or starts_with(n, "gpu::gen::"))
            continue;
        pw_ops.push_back(ins);
    }

    if(not pw_ops.empty())
    {
        auto insert_point = std::find_if(
            m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::gen::store"; });
        if(insert_point == m.end())
            insert_point = std::find_if(
                m.begin(), m.end(), [](const auto& ins) { return ins.name() == "@return"; });

        std::unordered_map<instruction_ref, instruction_ref> remap = load_map;
        remap.insert(index_remap.begin(), index_remap.end());
        for(auto ins : pw_ops)
        {
            std::vector<instruction_ref> new_inputs;
            for(auto input : ins->inputs())
            {
                if(contains(remap, input))
                    new_inputs.push_back(remap.at(input));
                else
                    new_inputs.push_back(input);
            }
            auto new_ins =
                m.insert_instruction(insert_point, ins->get_operator(), new_inputs);
            remap[ins] = new_ins;
        }

        // Rewire store to use the new scalar computation
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "gpu::gen::store")
                continue;
            auto inputs      = ins->inputs();
            auto value_input = inputs[2];
            if(contains(remap, value_input))
            {
                m.replace_instruction(
                    ins, ins->get_operator(), inputs[0], inputs[1], remap.at(value_input));
            }
        }
    }

    // Step 4: Lower gridwise_reduce to final gen IR reduction ops
    std::vector<instruction_ref> gw_reduces;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::gridwise_reduce")
            gw_reduces.push_back(ins);
    }

    for(auto gw_ins : gw_reduces)
    {
        auto v               = gw_ins->get_operator().to_value();
        auto rop             = v.at("op").to<std::string>();
        auto algo            = v.at("algo").to<std::string>();
        auto reduce_elements = v.at("reduce_elements").to<std::size_t>();
        auto block_size      = v.at("block_size").to<std::size_t>();
        auto input           = gw_ins->inputs().front();

        instruction_ref result;

        if(algo == "lane")
        {
            // Lane reduce: strided_load + lane_reduce
            // Each thread loads N strided elements and reduces locally
            auto loaded = m.insert_instruction(
                gw_ins,
                make_op("gpu::gen::strided_load",
                        {{"size", reduce_elements}, {"stride", std::size_t{1}}}),
                input,
                tid);
            result = m.insert_instruction(
                gw_ins, make_op("gpu::gen::lane_reduce", {{"op", rop}}), loaded);
        }
        else if(algo == "wave")
        {
            // Wave reduce: load + dpp_reduce
            auto loaded = m.insert_instruction(
                gw_ins, make_op("gpu::gen::load"), input, tid);
            result = m.insert_instruction(
                gw_ins, make_op("gpu::gen::dpp_reduce", {{"op", rop}}), loaded);
        }
        else // block
        {
            // Block reduce: strided_load + lane_reduce + dpp_reduce + reduce_waves
            // Each thread loads several strided elements, reduces locally,
            // then wave-level and block-level reductions follow.
            std::size_t elements_per_thread =
                (reduce_elements + block_size - 1) / block_size;
            if(elements_per_thread < 1)
                elements_per_thread = 1;

            instruction_ref partial;
            if(elements_per_thread > 1)
            {
                auto loaded = m.insert_instruction(
                    gw_ins,
                    make_op("gpu::gen::strided_load",
                            {{"size", elements_per_thread}, {"stride", block_size}}),
                    input,
                    tid);
                partial = m.insert_instruction(
                    gw_ins, make_op("gpu::gen::lane_reduce", {{"op", rop}}), loaded);
            }
            else
            {
                partial = m.insert_instruction(
                    gw_ins, make_op("gpu::gen::load"), input, tid);
            }

            // Wave-level reduction via DPP
            auto wave_result = m.insert_instruction(
                gw_ins, make_op("gpu::gen::dpp_reduce", {{"op", rop}}), partial);

            // Block-level reduction via LDS (lds_buffer added by blockwise pass)
            if(gw_ins->inputs().size() > 1)
            {
                auto lds_buf = gw_ins->inputs().back();
                result       = m.insert_instruction(
                    gw_ins, make_op("gpu::gen::reduce_waves", {{"op", rop}}), wave_result, lds_buf);
            }
            else
            {
                // Fallback if no LDS buffer (shouldn't happen for block algo)
                result = wave_result;
            }
        }

        m.replace_instruction(gw_ins, result);
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
