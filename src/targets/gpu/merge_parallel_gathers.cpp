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
#include <migraphx/gpu/merge_parallel_gathers.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_MERGE_PARALLEL_GATHERS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MERGE_PARALLEL_GATHERS)

namespace {

/**
 * Key for grouping gathers: (data_source, axis)
 */
struct gather_key
{
    instruction_ref data;
    int64_t axis;
    
    bool operator==(const gather_key& other) const
    {
        return data == other.data && axis == other.axis;
    }
};

struct gather_key_hash
{
    std::size_t operator()(const gather_key& k) const
    {
        return std::hash<instruction_ref>{}(k.data) ^ std::hash<int64_t>{}(k.axis);
    }
};

/**
 * Check if it's beneficial to merge gathers
 */
bool should_merge(const std::vector<instruction_ref>& gathers)
{
    // Need at least 2 gathers to merge
    if(gathers.size() < 2)
        return false;
    
    // Calculate total elements
    std::size_t total_elements = 0;
    for(const auto& gather_ins : gathers)
    {
        total_elements += gather_ins->get_shape().elements();
    }
    
    // Merging is beneficial when:
    // 1. Multiple small gathers (< 10K each) that can be batched
    // 2. Medium gathers (10K-100K) that benefit from reduced launches
    // 3. Not worth it for very large gathers (> 1M each) - may hurt cache
    
    std::size_t avg_size = total_elements / gathers.size();
    
    // Don't merge if individual gathers are already very large
    constexpr std::size_t too_large_threshold = 1000000;
    if(avg_size > too_large_threshold)
        return false;
    
    // Always beneficial for small gathers
    constexpr std::size_t small_threshold = 10000;
    if(avg_size < small_threshold)
        return true;
    
    // For medium gathers, need at least 3 to justify overhead
    if(gathers.size() >= 3)
        return true;
    
    return false;
}

/**
 * Merge a group of parallel gathers into a single gather + slices
 */
void merge_gather_group(module& m, const std::vector<instruction_ref>& gathers)
{
    if(!should_merge(gathers))
        return;
    
    // Get common properties
    auto ref_gather = gathers[0];
    auto ref_op = any_cast<op::gather>(ref_gather->get_operator());
    auto data_input = ref_gather->inputs()[0];
    auto axis = ref_op.axis;
    
    if(enabled(MIGRAPHX_TRACE_MERGE_PARALLEL_GATHERS{}))
    {
        std::cout << "Merging Parallel Gathers:\n";
        std::cout << "  Number of gathers: " << gathers.size() << "\n";
        std::cout << "  Gather axis: " << axis << "\n";
        std::cout << "  Data source: " << data_input->name() << "\n";
    }
    
    // Collect all indices and track sizes
    std::vector<instruction_ref> all_indices;
    std::vector<std::size_t> index_sizes;
    std::vector<std::size_t> cumulative_sizes;
    std::size_t total_size = 0;
    
    for(const auto& gather_ins : gathers)
    {
        auto indices_input = gather_ins->inputs()[1];
        auto indices_size = indices_input->get_shape().elements();
        
        all_indices.push_back(indices_input);
        index_sizes.push_back(indices_size);
        cumulative_sizes.push_back(total_size);
        total_size += indices_size;
    }
    
    // Insert concat of indices before first gather
    auto concat_axis = 0;  // Concat along first dimension
    auto concat_indices = m.insert_instruction(
        gathers[0],
        make_op("concat", {{"axis", concat_axis}}),
        all_indices);
    
    // Insert merged gather
    auto merged_gather = m.insert_instruction(
        gathers[0],
        ref_op,
        {data_input, concat_indices});
    
    if(enabled(MIGRAPHX_TRACE_MERGE_PARALLEL_GATHERS{}))
    {
        std::cout << "  Combined indices size: " << total_size << "\n";
        std::cout << "  Merged gather output: " << merged_gather->get_shape() << "\n";
    }
    
    // Replace each original gather with a slice of the merged result
    for(std::size_t i = 0; i < gathers.size(); ++i)
    {
        auto gather_ins = gathers[i];
        auto start = cumulative_sizes[i];
        auto end = start + index_sizes[i];
        
        // Create slice to extract this gather's portion
        // Slice along the axis dimension (typically axis 0 for indices)
        std::vector<int64_t> starts(merged_gather->get_shape().lens().size(), 0);
        std::vector<int64_t> ends(merged_gather->get_shape().lens().begin(),
                                   merged_gather->get_shape().lens().end());
        
        starts[axis] = start;
        ends[axis] = end;
        
        auto slice_ins = m.insert_instruction(
            std::next(merged_gather),
            make_op("slice", {{"starts", starts}, {"ends", ends}}),
            {merged_gather});
        
        // Replace original gather
        m.replace_instruction(gather_ins, slice_ins);
        
        if(enabled(MIGRAPHX_TRACE_MERGE_PARALLEL_GATHERS{}))
        {
            std::cout << "  Replaced gather " << i << " with slice [" << start << ":" << end << "]\n";
        }
    }
    
    if(enabled(MIGRAPHX_TRACE_MERGE_PARALLEL_GATHERS{}))
    {
        std::cout << "  Merge successful!\n\n";
    }
}

} // anonymous namespace

void merge_parallel_gathers::apply(module& m) const
{
    if(enabled(MIGRAPHX_DISABLE_MERGE_PARALLEL_GATHERS{}))
        return;
    
    // Group gathers by (data, axis)
    std::unordered_map<gather_key, std::vector<instruction_ref>, gather_key_hash> gather_groups;
    
    // Collect all gather operations
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "gather")
        {
            auto op = any_cast<op::gather>(ins->get_operator());
            auto data = ins->inputs()[0];
            
            gather_key key{data, op.axis};
            gather_groups[key].push_back(ins);
        }
    }
    
    // Merge each group that has multiple gathers
    for(auto& [key, gathers] : gather_groups)
    {
        if(gathers.size() >= 2)
        {
            merge_gather_group(m, gathers);
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

