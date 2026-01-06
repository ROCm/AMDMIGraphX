// fuse_horizontal.cpp - Horizontal fusion pass for parallel computation branches
#include <migraphx/fuse_horizontal.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Helper: Check if instruction is a reduce operation
static bool is_reduce(const instruction_ref& ins)
{
    const std::string& n = ins->name();
    return (n == "reduce_sum" or n == "reduce_mean" or n == "reduce_max" or 
            n == "reduce_min" or n == "reduce_prod");
}

// Helper: Check if instruction is a "view" operation (doesn't compute, just reshapes)
static bool is_view_op(const instruction_ref& ins)
{
    const std::string& n = ins->name();
    return (n == "multibroadcast" or n == "broadcast" or n == "reshape" or 
            n == "squeeze" or n == "unsqueeze" or n == "transpose" or
            n == "slice" or n == "contiguous" or n == "reshape_lazy");
}

// Helper: Check if instruction is a fuseable compute op (pointwise or reduce)
static bool is_fuseable_compute(const instruction_ref& ins)
{
    // Check pointwise attribute
    if(ins->get_operator().attributes().get("pointwise", false))
        return true;
    
    // Check if reduce
    if(is_reduce(ins))
        return true;
    
    return false;
}

// Helper: Get the "real" producer by tracing through view ops
static instruction_ref trace_through_views(instruction_ref ins)
{
    while(is_view_op(ins) && !ins->inputs().empty())
    {
        ins = ins->inputs().at(0);
    }
    return ins;
}

// Structure to represent a parallel branch from input to output
struct ParallelBranch
{
    instruction_ref input;      // The divergence point (e.g., @param)
    instruction_ref output;     // The convergence point (feeds into concat)
    std::vector<instruction_ref> ops;  // All ops in this branch (topological order)
};

// Trace a branch backwards from output to find the input parameter
// Returns the sequence of compute operations in the branch in topological order
static bool trace_branch_backwards(
    instruction_ref output,
    ParallelBranch& branch,
    std::unordered_set<instruction_ref>& visited)
{
    // First pass: collect all operations in this branch (backwards BFS)
    // Use a vector to maintain consistent ordering based on discovery
    std::vector<instruction_ref> worklist;
    std::vector<instruction_ref> branch_ops_vec;
    std::unordered_set<instruction_ref> branch_ops_set;
    instruction_ref main_input;
    
    worklist.push_back(output);
    
    while(!worklist.empty())
    {
        auto current = worklist.back();
        worklist.pop_back();
        
        if(!visited.insert(current).second)
            continue;
        
        // If this is a parameter, we found a potential input
        if(current->name() == "@param")
        {
            // Track the "main" input - pick the one with the largest shape
            // (the data input, not gamma/beta which are smaller)
            if(main_input == instruction_ref{})
            {
                main_input = current;
            }
            else
            {
                // Compare shapes - pick the one with more elements
                auto current_elements = current->get_shape().elements();
                auto main_elements = main_input->get_shape().elements();
                if(current_elements > main_elements)
                {
                    main_input = current;
                }
            }
            continue;
        }
        
        // If this is a compute op, add to our list (maintaining discovery order)
        if(is_fuseable_compute(current))
        {
            if(branch_ops_set.insert(current).second)
            {
                branch_ops_vec.push_back(current);
            }
        }
        
        // Trace through all inputs
        for(auto inp : current->inputs())
        {
            worklist.push_back(inp);
        }
    }
    
    if(main_input == instruction_ref{})
        return false;
    
    branch.input = main_input;
    branch.output = output;
    
    // Helper lambda to trace through view ops to find actual compute dependencies
    auto get_compute_dependencies = [&](instruction_ref op) -> std::vector<instruction_ref>
    {
        std::vector<instruction_ref> deps;
        std::vector<instruction_ref> worklist;
        std::unordered_set<instruction_ref> seen;
        
        for(auto inp : op->inputs())
        {
            worklist.push_back(inp);
        }
        
        while(!worklist.empty())
        {
            auto current = worklist.back();
            worklist.pop_back();
            
            if(!seen.insert(current).second)
                continue;
            
            // If this is a compute op in our branch, it's a dependency
            if(branch_ops_set.count(current) > 0)
            {
                deps.push_back(current);
            }
            else if(is_view_op(current))
            {
                // Trace through view ops to find the underlying compute
                for(auto inp : current->inputs())
                {
                    worklist.push_back(inp);
                }
            }
            // Otherwise it's a param/literal/external - not a compute dependency
        }
        
        return deps;
    };
    
    // Second pass: sort operations in topological order using Kahn's algorithm
    // Trace through view ops to find actual compute dependencies
    
    std::vector<instruction_ref> sorted_ops;
    std::unordered_set<instruction_ref> processed;
    
    // Use Kahn's algorithm with dependency-aware ordering
    while(sorted_ops.size() < branch_ops_vec.size())
    {
        bool made_progress = false;
        
        for(auto& op : branch_ops_vec)
        {
            if(processed.count(op) > 0)
                continue;
            
            // Check if all compute-op dependencies are already processed
            auto deps = get_compute_dependencies(op);
            bool all_inputs_ready = true;
            for(auto dep : deps)
            {
                if(processed.count(dep) == 0)
                {
                    all_inputs_ready = false;
                    break;
                }
            }
            
            if(all_inputs_ready)
            {
                sorted_ops.push_back(op);
                processed.insert(op);
                made_progress = true;
                break;  // Process one at a time for stable ordering
            }
        }
        
        if(!made_progress)
        {
            // Cycle detected or something wrong - just use remaining ops in order
            for(auto& op : branch_ops_vec)
            {
                if(processed.count(op) == 0)
                    sorted_ops.push_back(op);
            }
            break;
        }
    }
    
    branch.ops = sorted_ops;
    return true;
}

// Build a signature string for a branch (sequence of operation names)
static std::string build_branch_signature(const ParallelBranch& branch)
{
    std::string sig;
    for(auto& op : branch.ops)
    {
        if(!sig.empty())
            sig += ",";
        sig += op->name();
    }
    return sig;
}

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    
    std::cout << "[fuse_horizontal] Starting horizontal fusion pass" << std::endl;
    
    // Strategy: Find parallel branches that feed into concat operations
    // These are prime candidates for horizontal fusion
    
    // Step 1: Find all concat operations
    std::vector<instruction_ref> concats;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "concat")
        {
            concats.push_back(ins);
        }
    }
    
    std::cout << "[fuse_horizontal] Found " << concats.size() << " concat operations" << std::endl;
    
    // Step 2: For each concat, analyze its input branches
    for(auto concat_ins : concats)
    {
        auto concat_inputs = concat_ins->inputs();
        
        if(concat_inputs.size() < 2)
            continue;
        
        std::cout << "[fuse_horizontal] Analyzing concat with " << concat_inputs.size() 
                  << " inputs" << std::endl;
        
        // Trace each input back to find the parallel branches
        std::vector<ParallelBranch> branches;
        
        for(auto inp : concat_inputs)
        {
            ParallelBranch branch;
            std::unordered_set<instruction_ref> visited;
            
            // Trace through view ops to get the real compute output
            auto real_output = trace_through_views(inp);
            
            if(trace_branch_backwards(real_output, branch, visited))
            {
                branches.push_back(branch);
            }
        }
        
        std::cout << "[fuse_horizontal] Found " << branches.size() << " traceable branches" << std::endl;
        
        if(branches.size() < 2)
            continue;
        
        // Check if all branches have the same signature (isomorphic structure)
        std::string first_sig = build_branch_signature(branches[0]);
        bool all_same = true;
        
        for(size_t i = 1; i < branches.size(); i++)
        {
            std::string sig = build_branch_signature(branches[i]);
            if(sig != first_sig)
            {
                all_same = false;
                std::cout << "[fuse_horizontal] Branch " << i << " has different signature: " 
                          << sig << " vs " << first_sig << std::endl;
                break;
            }
        }
        
        if(!all_same)
        {
            std::cout << "[fuse_horizontal] Branches have different structures, skipping" << std::endl;
            continue;
        }
        
        std::cout << "[fuse_horizontal] All " << branches.size() 
                  << " branches have signature: " << first_sig << std::endl;
        
        // Check if input shapes match (required for batching)
        auto first_input_shape = branches[0].input->get_shape();
        bool shapes_match = true;
        
        for(size_t i = 1; i < branches.size(); i++)
        {
            if(branches[i].input->get_shape().lens() != first_input_shape.lens() ||
               branches[i].input->get_shape().type() != first_input_shape.type())
            {
                shapes_match = false;
                break;
            }
        }
        
        if(!shapes_match)
        {
            std::cout << "[fuse_horizontal] Input shapes don't match, skipping" << std::endl;
            continue;
        }
        
        // Check if all branches share the same main input instruction
        // If so, they have divergent computations from a common source that we can't batch
        bool all_same_input = true;
        for(size_t i = 1; i < branches.size(); i++)
        {
            if(branches[i].input != branches[0].input)
            {
                all_same_input = false;
                break;
            }
        }
        
        if(all_same_input)
        {
            std::cout << "[fuse_horizontal] All branches share the same input - divergent paths, skipping" << std::endl;
            continue;
        }
        
        std::cout << "[fuse_horizontal] Input shapes match, ready for batching!" << std::endl;
        std::cout << "[fuse_horizontal] Batching " << branches.size() 
                  << " parallel branches with " << branches[0].ops.size() << " ops each" << std::endl;
        
        // === IMPLEMENT BATCHING TRANSFORMATION ===
        
        // Step 1: Collect all input parameters
        std::vector<instruction_ref> input_params;
        for(auto& branch : branches)
        {
            input_params.push_back(branch.input);
        }
        
        // Step 2: Unsqueeze each input to add batch dimension at axis 0
        // {96, 262} -> {1, 96, 262}
        std::vector<instruction_ref> unsqueezed_inputs;
        for(auto inp : input_params)
        {
            auto unsq = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), inp);
            unsqueezed_inputs.push_back(unsq);
        }
        
        // Step 3: Concat along batch dimension to create batched input
        // {1, 96, 262} x 4 -> {4, 96, 262}
        auto batched_input = m.add_instruction(make_op("concat", {{"axis", 0}}), unsqueezed_inputs);
        
        std::cout << "[fuse_horizontal] Created batched input with shape {";
        auto batched_lens = batched_input->get_shape().lens();
        for(size_t i = 0; i < batched_lens.size(); i++)
        {
            if(i > 0) std::cout << ",";
            std::cout << batched_lens[i];
        }
        std::cout << "}" << std::endl;
        
        // Step 4: Replay the computation on the batched input
        // We need to map each original instruction to its batched version
        std::unordered_map<instruction_ref, instruction_ref> inst_map;
        
        // Map original inputs to the batched input
        // For each branch's input, create a mapping
        // But wait - all branches share the same batched input, so we need slices for ops
        // that reference the original input multiple times (like x - mean(x))
        
        // Actually, for LayerNorm, the input is used multiple times in the computation.
        // We can just use the batched_input directly since the operations are element-wise
        // or reduce along the correct axes.
        
        // Map all original input params to the batched input
        for(auto inp : input_params)
        {
            inst_map[inp] = batched_input;
        }
        
        // For each operation position, recreate it with batched inputs
        // We need to handle:
        // 1. Operations that only use the data input - just use batched version
        // 2. Operations that use constants/literals - need to broadcast to batch dim
        // 3. External inputs that DIFFER across branches - need to concat them
        // 4. Reduce operations - need to adjust axes by +1
        
        auto& first_branch = branches[0];
        
        for(size_t op_idx = 0; op_idx < first_branch.ops.size(); op_idx++)
        {
            auto& op = first_branch.ops[op_idx];
            std::string op_name = op->name();
            
            std::cout << "[fuse_horizontal] Replaying op " << op_idx << ": " << op_name 
                      << " with " << op->inputs().size() << " inputs" << std::endl;
            
            // Collect batched inputs for this operation
            std::vector<instruction_ref> batched_op_inputs;
            
            for(size_t inp_idx = 0; inp_idx < op->inputs().size(); inp_idx++)
            {
                auto orig_inp = op->inputs()[inp_idx];
                
                // Check if this input has a batched mapping
                if(inst_map.count(orig_inp) > 0)
                {
                    std::cout << "[fuse_horizontal]   Input " << inp_idx << ": found in map" << std::endl;
                    batched_op_inputs.push_back(inst_map[orig_inp]);
                }
                else
                {
                    std::cout << "[fuse_horizontal]   Input " << inp_idx << ": external (" 
                              << orig_inp->name() << ") - checking..." << std::endl;
                    
                    // First check: is this a view op wrapping a batched compute op?
                    // If so, recreate the view op on the batched output
                    bool handled_as_view = false;
                    if(is_view_op(orig_inp) && orig_inp->inputs().size() > 0)
                    {
                        // Trace through view ops to find the source
                        instruction_ref source = orig_inp->inputs()[0];
                        std::vector<instruction_ref> view_chain;
                        view_chain.push_back(orig_inp);
                        
                        while(is_view_op(source) && source->inputs().size() > 0)
                        {
                            view_chain.push_back(source);
                            source = source->inputs()[0];
                        }
                        
                        // If the source has been batched, recreate the view chain
                        if(inst_map.count(source) > 0)
                        {
                            std::cout << "[fuse_horizontal]     -> view op wrapping batched source" << std::endl;
                            auto batched_source = inst_map[source];
                            
                            // Recreate view ops in reverse order (innermost first)
                            instruction_ref current = batched_source;
                            for(auto it = view_chain.rbegin(); it != view_chain.rend(); ++it)
                            {
                                auto view_op = *it;
                                auto view_name = view_op->name();
                                auto view_val = view_op->get_operator().to_value();
                                
                                // Adjust the view op for batched shape
                                if(view_name == "multibroadcast")
                                {
                                    // Get the original broadcast shape and add batch dim
                                    auto orig_out_lens = view_val["out_lens"].to_vector<std::size_t>();
                                    std::vector<std::size_t> new_out_lens;
                                    new_out_lens.push_back(branches.size());
                                    new_out_lens.insert(new_out_lens.end(), orig_out_lens.begin(), orig_out_lens.end());
                                    
                                    current = m.add_instruction(
                                        make_op("multibroadcast", {{"out_lens", new_out_lens}}),
                                        current);
                                }
                                else
                                {
                                    // For other view ops, try to recreate with same params
                                    current = m.add_instruction(view_op->get_operator(), current);
                                }
                            }
                            
                            batched_op_inputs.push_back(current);
                            inst_map[orig_inp] = current;
                            handled_as_view = true;
                        }
                    }
                    
                    if(handled_as_view)
                        continue;
                    
                    // Not a view of batched compute - check if same across branches
                    bool inputs_same = true;
                    std::vector<instruction_ref> branch_inputs;
                    branch_inputs.push_back(orig_inp);
                    
                    for(size_t b = 1; b < branches.size(); b++)
                    {
                        if(op_idx < branches[b].ops.size() && 
                           inp_idx < branches[b].ops[op_idx]->inputs().size())
                        {
                            auto other_inp = branches[b].ops[op_idx]->inputs()[inp_idx];
                            branch_inputs.push_back(other_inp);
                            if(other_inp != orig_inp)
                            {
                                inputs_same = false;
                            }
                        }
                    }
                    
                    auto inp_shape = orig_inp->get_shape();
                    auto inp_lens = inp_shape.lens();
                    
                    // Get the expected batched shape for this input position
                    // by looking at what the corresponding batched op output would be
                    auto expected_ndim = batched_input->get_shape().ndim();
                    auto input_ndim = inp_lens.size();
                    
                    std::cout << "[fuse_horizontal]     -> inputs_same=" << inputs_same 
                              << ", branch_inputs.size()=" << branch_inputs.size() << std::endl;
                    
                    if(inputs_same)
                    {
                        // Same input across all branches - use multibroadcast
                        std::vector<std::size_t> broadcast_lens;
                        broadcast_lens.push_back(branches.size());
                        
                        // If input has fewer dims than batched input, add extra dims
                        for(size_t d = input_ndim; d < expected_ndim - 1; d++)
                        {
                            broadcast_lens.push_back(1);
                        }
                        broadcast_lens.insert(broadcast_lens.end(), inp_lens.begin(), inp_lens.end());
                        
                        auto unsq = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), orig_inp);
                        auto bcast = m.add_instruction(make_op("multibroadcast", {{"out_lens", broadcast_lens}}), unsq);
                        
                        batched_op_inputs.push_back(bcast);
                        inst_map[orig_inp] = bcast;
                    }
                    else
                    {
                        // Different inputs across branches - concat them
                        // First unsqueeze to add batch dimension
                        std::vector<instruction_ref> unsqueezed;
                        for(auto inp : branch_inputs)
                        {
                            auto unsq = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), inp);
                            unsqueezed.push_back(unsq);
                        }
                        auto concatenated = m.add_instruction(make_op("concat", {{"axis", 0}}), unsqueezed);
                        
                        // If input has fewer dims, we need to add dims to match batched shape
                        // e.g., gamma {262} -> {4, 262} but needs to be {4, 1, 262} for broadcast
                        instruction_ref result = concatenated;
                        if(input_ndim < expected_ndim - 1)
                        {
                            // Add missing dimensions via unsqueeze
                            // e.g., {4, 262} -> unsqueeze at axis 1 -> {4, 1, 262}
                            std::vector<int64_t> axes_to_add;
                            for(size_t d = 1; d <= expected_ndim - 1 - input_ndim; d++)
                            {
                                axes_to_add.push_back(static_cast<int64_t>(d));
                            }
                            result = m.add_instruction(make_op("unsqueeze", {{"axes", axes_to_add}}), concatenated);
                        }
                        
                        batched_op_inputs.push_back(result);
                        // Don't cache - different ops at this position use different inputs
                    }
                }
            }
            
            // Create the batched operation
            instruction_ref batched_op;
            
            if(is_reduce(op))
            {
                // For reduce ops, adjust axes by +1 to account for batch dimension
                auto op_val = op->get_operator().to_value();
                std::vector<int64_t> new_axes;
                
                if(op_val.contains("axes"))
                {
                    for(auto& ax : op_val["axes"])
                    {
                        new_axes.push_back(ax.to<int64_t>() + 1);  // Shift by 1
                    }
                }
                
                batched_op = m.add_instruction(
                    make_op(op_name, {{"axes", new_axes}}),
                    batched_op_inputs);
            }
            else
            {
                // For other ops (pointwise), just use the same operation
                batched_op = m.add_instruction(op->get_operator(), batched_op_inputs);
            }
            
            // Map this op from ALL branches to the batched version
            inst_map[op] = batched_op;
            for(size_t b = 1; b < branches.size(); b++)
            {
                if(op_idx < branches[b].ops.size())
                {
                    inst_map[branches[b].ops[op_idx]] = batched_op;
                }
            }
            
            std::cout << "[fuse_horizontal]   Created batched " << op_name << " with output shape {";
            auto out_lens = batched_op->get_shape().lens();
            for(size_t i = 0; i < out_lens.size(); i++)
            {
                if(i > 0) std::cout << ",";
                std::cout << out_lens[i];
            }
            std::cout << "}" << std::endl;
        }
        
        // Step 5: Slice the batched output and replace each branch's terminal
        // The batched output has shape {4, 96, 262} (batch on axis 0)
        // Each original branch terminal (e.g., sigmoid outputs) fed into unsqueeze->concat
        // We slice our batched output and replace those terminals
        
        auto batched_output = inst_map[first_branch.ops.back()];
        
        std::cout << "[fuse_horizontal] Batched output shape: {";
        auto out_lens = batched_output->get_shape().lens();
        for(size_t i = 0; i < out_lens.size(); i++)
        {
            if(i > 0) std::cout << ",";
            std::cout << out_lens[i];
        }
        std::cout << "}" << std::endl;
        
        // Get the original output shape (without batch dimension)
        std::vector<std::size_t> single_output_lens(out_lens.begin() + 1, out_lens.end());
        
        // For each branch, slice the batched output and replace the terminal
        for(size_t i = 0; i < branches.size(); i++)
        {
            // Slice out this branch's result: {4, 96, 262} -> {1, 96, 262}
            auto slice_i = m.add_instruction(
                make_op("slice", {{"axes", {0}},
                                  {"starts", {static_cast<int64_t>(i)}},
                                  {"ends", {static_cast<int64_t>(i + 1)}}}),
                batched_output);
            
            // Reshape to remove the batch dimension: {1, 96, 262} -> {96, 262}
            auto reshaped = m.add_instruction(
                make_op("reshape", {{"dims", single_output_lens}}),
                slice_i);
            
            // Replace the original terminal instruction for this branch
            auto original_terminal = branches[i].ops.back();
            m.replace_instruction(original_terminal, reshaped);
            
            std::cout << "[fuse_horizontal] Replaced branch " << i << " terminal with slice" << std::endl;
        }
        
        std::cout << "[fuse_horizontal] All branch terminals replaced" << std::endl;
    }
    
    // Sort and clean up
    m.sort();
    // Note: Don't run DCE here - let the pipeline's DCE handle it
    // This preserves our batched ops for debugging
    
    std::cout << "[fuse_horizontal] Horizontal fusion pass complete" << std::endl;
    std::cout << "[fuse_horizontal] Module after fusion:" << std::endl;
    m.debug_print();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
