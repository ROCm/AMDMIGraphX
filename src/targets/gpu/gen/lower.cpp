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

void gen_lower::apply(module& m) const
{
    // Collect copy instructions to lower
    std::vector<instruction_ref> copies_to_lower;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::copy")
        {
            copies_to_lower.push_back(ins);
        }
    }

    // Lower copy operations to vector_load/vector_store
    for(auto ins : copies_to_lower)
    {
        auto inputs = ins->inputs();
        if(inputs.size() != 2)
            continue;

        auto src_shape = inputs[0]->get_shape();
        auto dst_shape = inputs[1]->get_shape();

        // Determine vectorization
        std::vector<shape> shapes = {src_shape, dst_shape};
        auto normalized           = reduce_dims(normalize_permutation(shapes));
        auto axis                 = gpu::gen::find_fast_axis(normalized);
        auto vec_size             = compute_vector_size(normalized, axis);

        if(vec_size <= 1)
            vec_size = 1;

        // Insert global_id
        auto gid = m.insert_instruction(ins, make_op("gpu::gen::global_id"));

        // Insert vector_load from source
        auto load = m.insert_instruction(
            ins, make_op("gpu::gen::vector_load", {{"size", vec_size}}), inputs[0], gid);

        // Insert vector_store to destination
        auto store = m.insert_instruction(
            ins, make_op("gpu::gen::vector_store", {{"size", vec_size}}), inputs[1], gid, load);

        // Replace the copy with the store and remove the original copy
        m.replace_instruction(ins, store);
    }

    // Process remaining instructions
    for(auto ins : iterator_for(m))
    {
        // Lower pointwise operations to vector_load/vector_store
        if(ins->name() == "pointwise")
        {
            if(ins->module_inputs().empty())
                continue;

            auto inputs = ins->inputs();
            if(inputs.empty())
                continue;

            // Get input shapes
            std::vector<shape> input_shapes;
            for(auto input : inputs)
            {
                input_shapes.push_back(input->get_shape());
            }

            // Determine vectorization parameters
            auto normalized = reduce_dims(normalize_permutation(input_shapes));
            auto axis       = gpu::gen::find_fast_axis(normalized);
            auto vec_size   = compute_vector_size(normalized, axis);

            // If no vectorization possible, skip lowering for now
            // (the existing pointwise path will handle it)
            if(vec_size <= 1)
                continue;

            // Calculate number of elements per thread
            std::size_t total_elements      = input_shapes.front().elements();
            std::size_t elements_per_thread = vec_size;

            // Insert global_id to get thread index
            auto gid = m.insert_instruction(ins, make_op("gpu::gen::global_id"));

            // Insert vector_load for each input (except the output allocation)
            std::vector<instruction_ref> loaded_inputs;
            for(std::size_t i = 0; i < inputs.size() - 1; i++)
            {
                auto load = m.insert_instruction(
                    ins, make_op("gpu::gen::vector_load", {{"size", vec_size}}), inputs[i], gid);
                loaded_inputs.push_back(load);
            }

            // The inner pointwise module operates on the loaded vectors
            // For now, we keep the original pointwise and let code generation handle it
            // Future: inline the pointwise computation

            // Insert vector_store for the output
            // The output is the last input (allocation buffer)
            auto output_tensor = inputs.back();

            // Create a new pointwise that operates on vector inputs
            // For now, we mark this instruction as lowered and let codegen handle it
            // by adding metadata to the instruction

            (void)loaded_inputs;
            (void)output_tensor;
            (void)total_elements;
            (void)elements_per_thread;

            // Note: Full lowering would replace the pointwise instruction with:
            // 1. global_id
            // 2. vector_load for each input
            // 3. inline pointwise computation on vectors
            // 4. vector_store for output
            //
            // For now, the gen_pointwise_compiler handles this via arg transformers
        }

        // Lower tile_region operations
        if(ins->name() == "gpu::gen::tile_region")
        {
            // Get tile parameters via to_value()
            auto v         = ins->get_operator().to_value();
            auto tile_dims = v.at("tile_dims").to_vector<std::size_t>();
            auto axis      = v.at("axis").to<std::size_t>();

            // For now, tile lowering is handled during code generation
            // Future: expand tile into explicit loop structure with
            // global_id, local_id, and barrier operations
            (void)tile_dims;
            (void)axis;
        }
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
