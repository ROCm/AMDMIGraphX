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
#include <migraphx/gpu/gen/lower.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/reduce_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

// Determine the best vector size for a set of shapes
static std::size_t compute_vector_size(const std::vector<shape>& inputs, std::size_t axis)
{
    // Disable vectorization for fp8 types
    static const std::set<shape::type_t> fp8_types = {shape::fp8e4m3fn_type,
                                                      shape::fp8e4m3fnuz_type,
                                                      shape::fp8e5m2_type,
                                                      shape::fp8e5m2fnuz_type};
    if(std::any_of(inputs.begin(), inputs.end(), [&](const auto& s) {
           return contains(fp8_types, s.type());
       }))
        return 1;

    if(inputs.empty())
        return 1;

    auto ndims = inputs.front().ndim();
    if(axis >= ndims)
        return 1;

    // Check if all inputs have unit length on axis
    if(std::all_of(
           inputs.begin(), inputs.end(), [&](const auto& s) { return s.lens()[axis] == 1; }))
        return 1;

    // Find maximum vector size based on strides and alignment
    std::vector<std::size_t> max_vec_sizes;
    for(const auto& input : inputs)
    {
        auto stride = input.strides()[axis];
        auto len    = input.lens()[axis];

        if(not contains({0, 1}, stride))
        {
            max_vec_sizes.push_back(1);
            continue;
        }

        // Check alignment and find largest power of 2 that divides evenly
        std::size_t max_size = 1;
        for(std::size_t s : {8, 4, 2})
        {
            if(len % s == 0)
            {
                max_size = s;
                break;
            }
        }
        max_vec_sizes.push_back(max_size);
    }

    return *std::min_element(max_vec_sizes.begin(), max_vec_sizes.end());
}

// Lower a copy to vector_load/vector_store
static void lower_copy(
    module& m, instruction_ref ins, instruction_ref src, instruction_ref dst, std::size_t vec_size)
{
    // Determine if this is a tiled copy (inputs are tile_regions)
    bool is_tiled = (src->name() == "gpu::gen::tile_region");

    // Insert appropriate ID operation
    instruction_ref id_ins;
    if(is_tiled)
    {
        // Use local_id for within-tile indexing
        id_ins = m.insert_instruction(ins, make_op("gpu::gen::local_id"));
    }
    else
    {
        // Use global_id for 1D case
        id_ins = m.insert_instruction(ins, make_op("gpu::gen::global_id"));
    }

    // Insert vector_load from source
    auto load = m.insert_instruction(
        ins, make_op("gpu::gen::vector_load", {{"size", vec_size}}), src, id_ins);

    // Insert vector_store to destination
    auto store = m.insert_instruction(
        ins, make_op("gpu::gen::vector_store", {{"size", vec_size}}), dst, id_ins, load);

    // Replace the copy with the store
    m.replace_instruction(ins, store);
}

// Reduction operations mapping
static const std::unordered_map<std::string, std::string>& reduction_op_map()
{
    static const std::unordered_map<std::string, std::string> ops = {
        {"reduce_sum", "sum"},
        {"reduce_mean", "sum"}, // mean uses sum reduction then divides
        {"reduce_max", "max"},
        {"reduce_min", "min"},
        {"reduce_prod", "product"}};
    return ops;
}

// Check if an instruction is a reduction
static bool is_reduction(instruction_ref ins)
{
    return contains(reduction_op_map(), ins->name());
}

// Get the reduction type (sum, max, min, product)
static std::string get_reduction_type(const std::string& op_name)
{
    auto it = reduction_op_map().find(op_name);
    if(it != reduction_op_map().end())
        return it->second;
    return "sum";
}

// Lower a reduction to strided_load/accumulate/dpp_reduce/reduce_waves
static void lower_reduction(module& m, instruction_ref ins)
{
    auto reduce_type = get_reduction_type(ins->name());
    auto input       = ins->inputs().front();
    auto input_shape = input->get_shape();

    // Get reduction axes
    auto v    = ins->get_operator().to_value();
    auto axes = v.at("axes").to_vector<std::int64_t>();
    auto ndim = input_shape.ndim();

    // Normalize negative axes
    for(auto& axis : axes)
    {
        if(axis < 0)
            axis += ndim;
    }

    // Compute reduction elements
    std::size_t reduce_elements = 1;
    for(auto axis : axes)
    {
        reduce_elements *= input_shape.lens()[axis];
    }

    // Insert local_id for per-thread indexing
    auto local_id = m.insert_instruction(ins, make_op("gpu::gen::local_id"));

    // Compute elements per thread (strided access)
    // Each thread accumulates elements with a stride equal to workgroup size
    auto workgroup_size =
        m.insert_instruction(ins, make_op("gpu::gen::workgroup_size"));

    // Create zero constant for initial accumulator value
    auto zero_lit = m.add_literal(literal{shape{input_shape.type()}, {0}});

    // Strided load and accumulation loop
    // Each thread loads elements at indices: local_id, local_id + wg_size, local_id + 2*wg_size, ...
    auto strided_load = m.insert_instruction(
        ins,
        make_op("gpu::gen::strided_load"),
        input,
        local_id,
        local_id,           // base
        workgroup_size);    // stride

    // Accumulate the loaded value
    auto accumulated = m.insert_instruction(
        ins, make_op("gpu::gen::accumulate", {{"op", reduce_type}}), zero_lit, strided_load);

    // Wave-level reduction using DPP
    auto wave_reduced =
        m.insert_instruction(ins, make_op("gpu::gen::dpp_reduce", {{"op", reduce_type}}), accumulated);

    // Allocate LDS for cross-wave reduction (8 waves max)
    auto lds_shape = shape{input_shape.type(), {8}};
    auto lds =
        m.insert_instruction(ins, make_op("gpu::gen::lds_allocate", {{"shape", to_value(lds_shape)}}));

    // Block-level reduction across waves
    auto block_reduced = m.insert_instruction(
        ins, make_op("gpu::gen::reduce_waves", {{"op", reduce_type}}), wave_reduced, lds);

    // For reduce_mean, divide by number of elements
    if(ins->name() == "reduce_mean")
    {
        auto count_lit = m.add_literal(literal{shape{input_shape.type()}, {reduce_elements}});
        auto mean_result =
            m.insert_instruction(ins, make_op("div"), block_reduced, count_lit);
        m.replace_instruction(ins, mean_result);
    }
    else
    {
        m.replace_instruction(ins, block_reduced);
    }
}

void gen_lower::apply(module& m) const
{
    // Collect instructions to lower
    std::vector<instruction_ref> copies_to_lower;
    std::vector<instruction_ref> reductions_to_lower;

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::copy")
        {
            copies_to_lower.push_back(ins);
        }
        else if(is_reduction(ins))
        {
            reductions_to_lower.push_back(ins);
        }
    }

    // Lower copy operations
    for(auto ins : copies_to_lower)
    {
        auto inputs = ins->inputs();
        if(inputs.size() != 2)
            continue;

        auto src       = inputs[0];
        auto dst       = inputs[1];
        auto src_shape = src->get_shape();
        auto dst_shape = dst->get_shape();

        // Determine vectorization
        std::vector<shape> shapes = {src_shape, dst_shape};
        auto normalized           = reduce_dims(normalize_permutation(shapes));
        auto axis                 = gpu::gen::find_fast_axis(normalized);
        auto vec_size             = compute_vector_size(normalized, axis);

        if(vec_size <= 1)
            vec_size = 1;

        // Lower copy to vector_load/vector_store
        lower_copy(m, ins, src, dst, vec_size);
    }

    // Lower reduction operations
    for(auto ins : reductions_to_lower)
    {
        lower_reduction(m, ins);
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
