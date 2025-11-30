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

// Check if a shape needs LDS copy (non-contiguous on fast axis)
static bool needs_lds_copy(const shape& s, std::size_t fast_axis)
{
    if(s.ndim() == 0)
        return false;
    if(fast_axis >= s.ndim())
        return false;

    // If stride on fast axis is not 0 or 1, it's strided/transposed
    auto stride = s.strides()[fast_axis];
    return stride != 0 && stride != 1;
}

// Compute LDS shape with padding to avoid bank conflicts
static shape compute_lds_shape(const shape& s)
{
    auto lens    = s.lens();
    auto strides = s.strides();

    // Add 1 element padding on the last dimension to avoid bank conflicts
    if(not strides.empty())
    {
        strides.back() = strides.back() + 1;
    }

    return shape{s.type(), lens, strides};
}

// Compute tile dimensions for a given shape
static std::vector<std::size_t> compute_tile_dims(const shape& s, std::size_t tile_axis)
{
    std::vector<std::size_t> tile_dims;
    auto lens = s.lens();

    // Start from tile_axis to the end
    for(std::size_t i = tile_axis; i < lens.size(); i++)
    {
        // Use reasonable tile sizes - these can be tuned
        std::size_t tile_size = 32;
        if(i == lens.size() - 1)
        {
            // Last dimension: use 64 for better vectorization
            tile_size = 64;
        }
        tile_dims.push_back(std::min(tile_size, lens[i]));
    }

    return tile_dims;
}

// Lower a simple 1D copy to vector_load/vector_store
static void lower_simple_copy(
    module& m, instruction_ref ins, instruction_ref src, instruction_ref dst, std::size_t vec_size)
{
    // Insert global_id for 1D case
    auto gid = m.insert_instruction(ins, make_op("gpu::gen::global_id"));

    // Insert vector_load from source
    auto load =
        m.insert_instruction(ins, make_op("gpu::gen::vector_load", {{"size", vec_size}}), src, gid);

    // Insert vector_store to destination
    auto store = m.insert_instruction(
        ins, make_op("gpu::gen::vector_store", {{"size", vec_size}}), dst, gid, load);

    // Replace the copy with the store
    m.replace_instruction(ins, store);
}

// Lower a tiled multi-dimensional copy
static void lower_tiled_copy(module& m,
                             instruction_ref ins,
                             instruction_ref src,
                             instruction_ref dst,
                             const std::vector<std::size_t>& tile_dims,
                             std::size_t tile_axis,
                             std::size_t vec_size)
{
    // Insert workgroup_id
    auto wg_id = m.insert_instruction(ins, make_op("gpu::gen::workgroup_id"));

    // Create tile_region for source
    auto src_tile = m.insert_instruction(
        ins,
        make_op("gpu::gen::tile_region", {{"tile_dims", tile_dims}, {"axis", tile_axis}}),
        src,
        wg_id);

    // Create tile_region for destination
    auto dst_tile = m.insert_instruction(
        ins,
        make_op("gpu::gen::tile_region", {{"tile_dims", tile_dims}, {"axis", tile_axis}}),
        dst,
        wg_id);

    // Insert local_id for within-tile indexing
    auto lid = m.insert_instruction(ins, make_op("gpu::gen::local_id"));

    // Insert vector_load from source tile
    auto load = m.insert_instruction(
        ins, make_op("gpu::gen::vector_load", {{"size", vec_size}}), src_tile, lid);

    // Insert vector_store to destination tile
    auto store = m.insert_instruction(
        ins, make_op("gpu::gen::vector_store", {{"size", vec_size}}), dst_tile, lid, load);

    // Replace the copy with the store
    m.replace_instruction(ins, store);
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

        // Get schedule from operation
        auto v        = ins->get_operator().to_value();
        auto schedule = v.get("schedule", std::string("per_thread"));

        // Determine vectorization
        std::vector<shape> shapes = {src_shape, dst_shape};
        auto normalized           = reduce_dims(normalize_permutation(shapes));
        auto axis                 = gpu::gen::find_fast_axis(normalized);
        auto vec_size             = compute_vector_size(normalized, axis);

        if(vec_size <= 1)
            vec_size = 1;

        // Check if this is a multi-dimensional tensor that needs tiling
        bool needs_tiling = src_shape.ndim() > 1 && schedule == "per_block";

        if(needs_tiling)
        {
            // Determine tile axis (usually 1 for batch dims)
            std::size_t tile_axis = 1;
            if(src_shape.ndim() <= 1)
                tile_axis = 0;

            auto tile_dims = compute_tile_dims(src_shape, tile_axis);
            lower_tiled_copy(m, ins, src, dst, tile_dims, tile_axis, vec_size);
        }
        else
        {
            // Simple 1D lowering
            lower_simple_copy(m, ins, src, dst, vec_size);
        }
    }

    // Process tile_region operations (currently just for validation)
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gpu::gen::tile_region")
        {
            // Get tile parameters via to_value()
            auto v         = ins->get_operator().to_value();
            auto tile_dims = v.at("tile_dims").to_vector<std::size_t>();
            auto axis      = v.at("axis").to<std::size_t>();

            // For now, tile_region is kept in IR and handled during code generation
            (void)tile_dims;
            (void)axis;
        }
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
