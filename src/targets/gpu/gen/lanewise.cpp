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

    // Step 2: Insert loads for input params and rebuild pointwise ops
    // with scalar inputs. We find all params (except z_output) that are
    // used by non-gen ops, insert loads, and rebuild those ops.
    std::unordered_map<instruction_ref, instruction_ref> load_map;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@param")
            continue;
        auto pname = ins->get_operator().to_value().at("parameter").to<std::string>();
        if(pname == "z_output")
            continue;

        bool needs_load = false;
        for(auto out : ins->outputs())
        {
            auto n = out->name();
            if(starts_with(n, "@") or starts_with(n, "gpu::gen::"))
                continue;
            needs_load = true;
            break;
        }
        if(not needs_load)
            continue;

        auto load     = m.insert_instruction(std::next(tid), make_op("gpu::gen::load"), ins, tid);
        load_map[ins] = load;
    }

    // Rebuild pointwise ops with all scalar inputs at once
    for(auto ins : iterator_for(m))
    {
        auto n = ins->name();
        if(starts_with(n, "@") or starts_with(n, "gpu::gen::"))
            continue;

        auto inputs       = ins->inputs();
        bool any_replaced = false;
        std::vector<instruction_ref> new_inputs;
        for(auto input : inputs)
        {
            if(contains(load_map, input))
            {
                new_inputs.push_back(load_map.at(input));
                any_replaced = true;
            }
            else
            {
                new_inputs.push_back(input);
            }
        }
        if(any_replaced)
        {
            m.replace_instruction(ins, ins->get_operator(), new_inputs);
        }
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
