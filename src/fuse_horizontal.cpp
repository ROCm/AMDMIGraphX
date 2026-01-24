// fuse_horizontal.cpp - Horizontal fusion pass for parallel operations
// Uses level-order reordering to identify and batch parallel operations

#include <migraphx/fuse_horizontal.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Check if an op can be batched
static bool is_batchable_op(const std::string& name)
{
    static const std::unordered_set<std::string> batchable_ops = {
        // Pointwise ops
        "add", "sub", "mul", "div", "pow", "sqrt", "rsqrt",
        "exp", "log", "abs", "neg", "relu", "sigmoid", "tanh",
        "erf", "gelu", "silu", "floor", "ceil", "round",
        // Reduce ops
        "reduce_mean", "reduce_sum", "reduce_max", "reduce_min"
    };
    return batchable_ops.count(name) > 0;
}

// Get a signature for an instruction (for grouping similar ops)
static std::string get_op_signature(instruction_ref ins)
{
    auto shape = ins->get_shape();
    std::string sig = ins->name() + "|";
    for(auto d : shape.lens())
        sig += std::to_string(d) + ",";
    return sig;
}

// Reorder instructions to level-order (parallel ops grouped together)
static void reorder_to_level_order(module& m)
{
    std::cout << "[fuse_horizontal] Reordering to level-order..." << std::endl;
    
    // Assign levels to each instruction
    std::unordered_map<const instruction*, size_t> levels;
    
    for(auto it : iterator_for(m))
    {
        size_t max_input_level = 0;
        for(auto inp : it->inputs())
        {
            auto inp_ptr = std::addressof(*inp);
            if(levels.count(inp_ptr) > 0)
            {
                max_input_level = std::max(max_input_level, levels[inp_ptr]);
            }
        }
        levels[std::addressof(*it)] = max_input_level + 1;
    }
    
    // Collect instructions by level
    std::vector<std::vector<instruction_ref>> by_level;
    for(auto it : iterator_for(m))
    {
        size_t level = levels[std::addressof(*it)];
        if(level >= by_level.size())
            by_level.resize(level + 1);
        by_level[level].push_back(it);
    }
    
    std::cout << "[fuse_horizontal] Assigned " << m.size() << " instructions to " 
              << by_level.size() << " levels" << std::endl;
    
    // Reorder by moving instructions level by level
    auto end_it = m.end();
    for(size_t lvl = 0; lvl < by_level.size(); lvl++)
    {
        for(auto ins : by_level[lvl])
        {
            m.move_instruction(ins, end_it);
        }
    }
    
    std::cout << "[fuse_horizontal] Level-order reordering complete" << std::endl;
}

void fuse_horizontal::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    
    std::cout << "[fuse_horizontal] Starting horizontal fusion pass" << std::endl;
    
    // Print IR BEFORE reordering
    std::cout << "\n=== IR BEFORE LEVEL-ORDER ===" << std::endl;
    m.debug_print();
    std::cout << "=== END IR BEFORE LEVEL-ORDER ===\n" << std::endl;
    
    // Step 1: Reorder to level-order
    reorder_to_level_order(m);
    
    // Print IR AFTER reordering
    std::cout << "\n=== IR AFTER LEVEL-ORDER ===" << std::endl;
    m.debug_print();
    std::cout << "=== END IR AFTER LEVEL-ORDER ===\n" << std::endl;
    
    // Step 2: Find ops at the same level with same signature
    // Only batch ops that are truly parallel (same level, no interdependencies)
    
    // Assign levels to each instruction
    std::unordered_map<const instruction*, size_t> levels;
    for(auto it : iterator_for(m))
    {
        size_t max_input_level = 0;
        for(auto inp : it->inputs())
        {
            auto inp_ptr = std::addressof(*inp);
            if(levels.count(inp_ptr) > 0)
            {
                max_input_level = std::max(max_input_level, levels[inp_ptr]);
            }
        }
        levels[std::addressof(*it)] = max_input_level + 1;
    }
    
    // Build a list of all batchable instructions with their signatures AND levels
    std::vector<std::tuple<instruction_ref, std::string, size_t>> all_ops;
    for(auto it : iterator_for(m))
    {
        if(is_batchable_op(it->name()))
        {
            size_t level = levels[std::addressof(*it)];
            all_ops.push_back({it, get_op_signature(it), level});
        }
    }
    
    std::cout << "[fuse_horizontal] Found " << all_ops.size() << " batchable ops" << std::endl;
    
    // Group by signature AND level (only batch ops at the same level)
    std::map<std::pair<std::string, size_t>, std::vector<instruction_ref>> groups;
    for(auto& [ins, sig, level] : all_ops)
    {
        groups[{sig, level}].push_back(ins);
    }
    
    // Filter to groups with >= 2 ops
    int batch_count = 0;
    for(auto& [key, ops] : groups)
    {
        if(ops.size() < 2)
            continue;
        
        auto& [sig, level] = key;
        std::cout << "[fuse_horizontal] Group [" << sig << " @level=" << level << "]: " << ops.size() << " ops" << std::endl;
        
        // Check if all ops have the same number of inputs
        size_t num_inputs = ops[0]->inputs().size();
        bool compatible = true;
        for(size_t i = 1; i < ops.size() && compatible; i++)
        {
            if(ops[i]->inputs().size() != num_inputs)
                compatible = false;
        }
        
        if(!compatible)
        {
            std::cout << "[fuse_horizontal]   Skipping - incompatible input counts" << std::endl;
            continue;
        }
        
        // Check that none of the ops have each other as inputs (must be truly parallel)
        std::unordered_set<const instruction*> ops_set;
        for(auto op : ops)
            ops_set.insert(std::addressof(*op));
        
        bool has_internal_dep = false;
        for(auto op : ops)
        {
            for(auto inp : op->inputs())
            {
                if(ops_set.count(std::addressof(*inp)) > 0)
                {
                    has_internal_dep = true;
                    break;
                }
            }
            if(has_internal_dep) break;
        }
        
        if(has_internal_dep)
        {
            std::cout << "[fuse_horizontal]   Skipping - ops have internal dependencies" << std::endl;
            continue;
        }
        
        // For each input position, check if shapes are compatible for concat
        bool shapes_ok = true;
        for(size_t inp_idx = 0; inp_idx < num_inputs && shapes_ok; inp_idx++)
        {
            auto first_shape = ops[0]->inputs()[inp_idx]->get_shape().lens();
            for(size_t i = 1; i < ops.size() && shapes_ok; i++)
            {
                auto shape = ops[i]->inputs()[inp_idx]->get_shape().lens();
                if(shape != first_shape)
                    shapes_ok = false;
            }
        }
        
        if(!shapes_ok)
        {
            std::cout << "[fuse_horizontal]   Skipping - incompatible input shapes" << std::endl;
            continue;
        }
        
        // Now batch this group
        std::cout << "[fuse_horizontal]   Batching " << ops.size() << " ops" << std::endl;
        
        // For each input position, create batched input
        std::vector<instruction_ref> batched_inputs;
        for(size_t inp_idx = 0; inp_idx < num_inputs; inp_idx++)
        {
            std::vector<instruction_ref> unsqueezed;
            for(auto op : ops)
            {
                auto unsq = m.add_instruction(
                    make_op("unsqueeze", {{"axes", {0}}}), 
                    op->inputs()[inp_idx]);
                unsqueezed.push_back(unsq);
            }
            auto batched = m.add_instruction(
                make_op("concat", {{"axis", 0}}), 
                unsqueezed);
            batched_inputs.push_back(batched);
        }
        
        // Create the batched op
        instruction_ref batched_op;
        std::string op_name = ops[0]->name();
        
        // For reduce ops, we need to shift the axis by 1 because we added a batch dimension
        auto adjusted_op = ops[0]->get_operator();
        if(op_name.find("reduce") != std::string::npos)
        {
            auto val = adjusted_op.to_value();
            if(val.contains("axes"))
            {
                auto axes = val["axes"].to_vector<int64_t>();
                for(auto& ax : axes)
                    ax += 1;  // Shift axis by 1 for batch dimension
                val["axes"] = axes;
                adjusted_op.from_value(val);
            }
        }
        
        // Handle different arities
        if(batched_inputs.size() == 1)
        {
            batched_op = m.add_instruction(adjusted_op, batched_inputs[0]);
        }
        else if(batched_inputs.size() == 2)
        {
            batched_op = m.add_instruction(adjusted_op, batched_inputs[0], batched_inputs[1]);
        }
        else
        {
            std::cout << "[fuse_horizontal]   Skipping - unsupported arity " << batched_inputs.size() << std::endl;
            continue;
        }
        
        // Slice output and replace each original op
        auto orig_shape = ops[0]->get_shape().lens();
        for(size_t i = 0; i < ops.size(); i++)
        {
            auto sliced = m.add_instruction(
                make_op("slice", {{"axes", {0}},
                                  {"starts", {static_cast<int64_t>(i)}},
                                  {"ends", {static_cast<int64_t>(i + 1)}}}),
                batched_op);
            
            auto reshaped = m.add_instruction(
                make_op("reshape", {{"dims", orig_shape}}),
                sliced);
            
            m.replace_instruction(ops[i], reshaped);
        }
        
        batch_count++;
    }
    
    std::cout << "[fuse_horizontal] Batched " << batch_count << " groups" << std::endl;
    
    // Sort to fix any ordering issues
    std::cout << "[fuse_horizontal] Sorting module..." << std::endl;
    m.sort();
    
    // Run DCE to clean up
    std::cout << "[fuse_horizontal] Running DCE..." << std::endl;
    dead_code_elimination{}.apply(m);
    
    std::cout << "after pass" << std::endl;
    m.debug_print();
    
    std::cout << "[fuse_horizontal] Horizontal fusion complete" << std::endl;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
