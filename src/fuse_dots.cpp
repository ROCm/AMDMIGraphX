// fuse_dots.cpp
#include <migraphx/fuse_dots.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {


static bool is_dot_like(const instruction_ref& ins)
{
    const std::string& n = ins->name();
    return (n == "dot" or n == "quant_dot");
}

// Extract (M,K) from lhs shape using the last two dims.
static bool get_mk(const shape& lhs, std::size_t& M, std::size_t& K)
{
    if(lhs.lens().size() < 2) return false;
    const auto& l = lhs.lens();
    M = l[l.size() - 2];
    K = l[l.size() - 1];
    return true;
}

inline bool comes_before(const module& m, instruction_ref a, instruction_ref b)
{
    return (std::distance(m.begin(), a) < std::distance(m.begin(), b));
}

// Clone 'ins' and its producer chain so that the returned instruction
// is guaranteed to be before 'insert_pos'.
// - Reuses instructions already before insert_pos.
// - Memoizes cloned nodes.
// - Parameters are reused.
// - Literals are reused (optionally: move or clone as needed).
inline instruction_ref clone_before(module& m,
                                    instruction_ref ins,
                                    instruction_ref insert_pos,
                                    std::unordered_map<instruction_ref, instruction_ref>& memo)
{
    // Already cloned?
    if(auto it = memo.find(ins); it != memo.end())
        return it->second;

    // If it's already before insert_pos, reuse it
    if(comes_before(m, ins, insert_pos))
    {
        memo[ins] = ins;
        return ins;
    }

    // Parameters: reuse
    if(ins->name() == "@param")
    {
        memo[ins] = ins;
        return ins;
    }

    // TODO check if strict ordering is neeeded
    if(ins->name() == "@literal")
    {
        memo[ins] = ins;
        return ins;
    }

    // Recursively ensure all inputs are before insert_pos
    std::vector<instruction_ref> new_inputs;
    new_inputs.reserve(ins->inputs().size());
    for(auto in : ins->inputs())
    {
        new_inputs.push_back(clone_before(m, in, insert_pos, memo));
    }

    // Recreate the instruction before insert_pos with the same operator
    auto new_ins = m.insert_instruction(insert_pos, ins->get_operator(), new_inputs);
    m.replace_instruction(ins, new_ins);

    memo[ins] = new_ins;
    return new_ins;
}

void reorder_by_level(module& m, const std::unordered_map<instruction_ref, int>& level)
{
    // Build buckets
    std::map<int, std::vector<instruction_ref>> buckets;
    for(auto ins : iterator_for(m))
    {
        buckets[level.at(ins)].push_back(ins);
    }
        
    // New order: concat buckets by ascending level
    std::vector<instruction_ref> new_order;
    new_order.reserve(m.size());
    for(auto& kv : buckets)
    {
        new_order.insert(new_order.end(), kv.second.begin(), kv.second.end());
    }

    // Physically reorder
    auto it = m.begin();
    for(auto ins : new_order)
    {
        if(it != ins)
        {
            m.move_instruction(ins, it);
        }
        else
            it++;
        
    }
}

template <class T>
static void fuse_groups(module_pass_manager& mpm, 
                        const std::map<T, std::vector<instruction_ref>>& groups, size_t fuse_threshold)
{
    auto& m = mpm.get_module();
    for(auto& gkv : groups)
    {
        auto& group = gkv.second;
        if(group.size() < fuse_threshold)
            continue; // below threshold, skip

        // m.debug_print();
        
        std::string op_name = group.front()->name();
        size_t num_inputs = group.front()->inputs().size();

        // 5a) Determine insertion point: use the first instruction in program order
        instruction_ref insert_pos = group.back();
        for(auto ins : group)
        {
            // std::cout << "group_ins: " << std::endl;
            // m.debug_print(ins);
            if(comes_before(m, ins, insert_pos))
                insert_pos = ins;
        }

        // 5b) Prepare unsqueezed inputs
        std::vector<instruction_ref> lhs_unsqueezed;
        std::vector<instruction_ref> rhs_unsqueezed;
        lhs_unsqueezed.reserve(group.size());
        rhs_unsqueezed.reserve(group.size());

        std::unordered_map<instruction_ref, instruction_ref> memo;


        for(auto ins : group)
        {
            auto lhs = ins->inputs().at(0);
            // Ensure producer chains are before insert_pos
            auto lhs_b = clone_before(m, lhs, insert_pos, memo);
            // Now create the unsqueeze ops at insert_pos using the "before" versions
            auto lhs_u = m.insert_instruction(
                insert_pos, make_op("unsqueeze", {{"axes", {0}}}), lhs_b);
            lhs_unsqueezed.push_back(lhs_u);
            if(num_inputs == 2)
            {
                auto rhs = ins->inputs().at(1);
                auto rhs_b = clone_before(m, rhs, insert_pos, memo);
            
                auto rhs_u = m.insert_instruction(
                    insert_pos, make_op("unsqueeze", {{"axes", {0}}}), rhs_b);

            
                rhs_unsqueezed.push_back(rhs_u);
            }
            
        }
        instruction_ref fused;

        // 5c) Concat along new batch axis (axis=0)
        auto lhs_cat = m.insert_instruction(
            insert_pos, make_op("concat", {{"axis", 0}}), lhs_unsqueezed);
        // Optional: ensure contiguous layout if needed
        lhs_cat = m.insert_instruction(insert_pos, make_op("contiguous"), lhs_cat);
        if(num_inputs == 1)
        {
            // 5d) Single batched op
            fused = m.insert_instruction(insert_pos, make_op(op_name), lhs_cat);
        }
        else
        {
            auto rhs_cat = m.insert_instruction(
                insert_pos, make_op("concat", {{"axis", 0}}), rhs_unsqueezed);
            
            rhs_cat = m.insert_instruction(insert_pos, make_op("contiguous"), rhs_cat);

            fused = m.insert_instruction(insert_pos, make_op(op_name), lhs_cat, rhs_cat);
        }
        

        // 5e) Slice per batch and squeeze axis 0, then replace originals
        // To keep outputs ordered deterministically, use the order in 'group'
        for(std::size_t i = 0; i < group.size(); ++i)
        {
            auto slice_i = m.insert_instruction(
                insert_pos,
                make_op("slice",
                        {{"axes", {0}},
                            {"starts", {static_cast<int64_t>(i)}},
                            {"ends", {static_cast<int64_t>(i + 1)}}}),
                fused);

            std::vector<size_t> slice_lens = slice_i->get_shape().lens();
            slice_lens.erase(slice_lens.begin());

            auto out_i = m.insert_instruction(
                insert_pos, make_op("reshape", {{"dims", slice_lens}}), slice_i);
            

            // Replace original dot with its fused output
            m.replace_instruction(group[i], out_i);
            // std::cout << "replaced ins: " << std::endl;
            // m.debug_print(out_i);
        }
        mpm.run_pass(dead_code_elimination{});
        mpm.run_pass(simplify_reshapes{}); // eliminate possible duplicate slice/concat patterns
        // mpm.run_pass(simplify_algebra{});
        // std::cout << "new program" << std::endl;
        // m.debug_print();
    }
}

static void fuse_dot_ins(module_pass_manager& mpm, const std::unordered_map<instruction_ref, int>& level, size_t fuse_threshold) {
    auto& m = mpm.get_module();
    // 4) Collect dot-like ops by layer
    std::map<int, std::vector<instruction_ref>> dot_layers;
    for(auto ins : iterator_for(m))
    {
        if(is_dot_like(ins))
        {
            int lvl = 0;
            auto it = level.find(ins);
            if(it != level.end())
                lvl = it->second;
            dot_layers[lvl].push_back(ins);
        }
    }

    // 5) For each layer, group by (dtype, output lens, M, K) and fuse if group size >= threshold    

    for(const auto& kv : dot_layers)
    {
        const auto& nodes = kv.second;

        using key_t = std::tuple<shape::type_t, std::vector<std::size_t>, std::size_t, std::size_t>;

        std::map<key_t, std::vector<instruction_ref>> groups;
        for(auto ins : nodes)
        {
            const auto& out_s = ins->get_shape();
            const auto& lhs_s = ins->inputs().at(0)->get_shape();
            std::size_t M = 0, K = 0;
            if(!get_mk(lhs_s, M, K))
                continue;

            key_t k = std::make_tuple(out_s.type(), out_s.lens(), M, K);
            groups[k].push_back(ins);
        }

        fuse_groups(mpm, groups, fuse_threshold);
    }
}

static void fuse_pw_ins(module_pass_manager& mpm, const std::unordered_map<instruction_ref, int>& level, size_t fuse_threshold) {
    auto& m = mpm.get_module();
    // 4) Collect pointwise ops by layer
    std::map<int, std::vector<instruction_ref>> pw_layers;
    for(auto ins : iterator_for(m))
    {
        auto op = ins->get_operator();
        if(op.attributes().contains("pointwise"))
        {
            int lvl = 0;
            auto it = level.find(ins);
            if(it != level.end())
                lvl = it->second;
            pw_layers[lvl].push_back(ins);
        }
    }

    // 5) For each layer, group by (dtype, output lens) and fuse if group size >= threshold    

    for(const auto& kv : pw_layers)
    {
        const auto& nodes = kv.second;

        using key_t = std::tuple<shape::type_t, std::vector<size_t>, std::string>;

        std::map<key_t, std::vector<instruction_ref>> groups;
        for(auto ins : nodes)
        {
            const auto& out_s = ins->get_shape();

            key_t k = std::make_tuple(out_s.type(), out_s.lens(), ins->name());
            groups[k].push_back(ins);
        }

        fuse_groups(mpm, groups, fuse_threshold);
    }
}

void fuse_dots::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    // configurable threshold
    size_t fuse_threshold = 40;
    // 1) Build indegree map and initialize levels
    std::unordered_map<instruction_ref, std::size_t> indeg;
    indeg.reserve(std::distance(m.begin(), m.end()));
    std::unordered_map<instruction_ref, int> level;
    level.reserve(indeg.size());

    for(auto ins : iterator_for(m))
    {
        indeg[ins] = ins->inputs().size();
    }

    // 2) Kahn’s algorithm queue initialization
    std::queue<instruction_ref> q;
    for(auto ins : iterator_for(m))
    {
        if(indeg.at(ins) == 0)
        {
            level[ins] = 0;
            q.push(ins);
        }
    }

    // 3) Forward BFS/toposort computing levels
    while(!q.empty())
    {
        auto u = q.front();
        q.pop();
        int ulevel = level[u];

        for(auto v : u->outputs())
        {
            // Level is 1 + max(parent levels)
            auto it = level.find(v);
            int cand = ulevel + 1;
            if(it == level.end())
                level[v] = cand;
            else
                it->second = std::max(it->second, cand);

            // Decrement indegree and enqueue when ready
            auto dit = indeg.find(v);
            if(dit != indeg.end() && dit->second > 0)
            {
                dit->second -= 1;
                if(dit->second == 0)
                    q.push(v);
            }
        }
    }

    reorder_by_level(m, level);    

    fuse_dot_ins(mpm, level, fuse_threshold);
    
    fuse_pw_ins(mpm, level, fuse_threshold);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
