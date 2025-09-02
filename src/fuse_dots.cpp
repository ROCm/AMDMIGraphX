// fuse_dots.cpp
#include <migraphx/fuse_dots.hpp>
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

void fuse_dots::apply(module& m) const
{
    // configurable threshold
    size_t fuse_threshold = 10;
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
    using key_t = std::tuple<shape::type_t, std::vector<std::size_t>, std::size_t, std::size_t>;

    for(const auto& kv : dot_layers)
    {
        const auto& nodes = kv.second;

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

        for(auto& gkv : groups)
        {
            auto& group = gkv.second;
            if(group.size() < fuse_threshold)
                continue; // below threshold, skip

            // 5a) Determine insertion point: use the first instruction in program order
            instruction_ref insert_pos = group.front();
            for(auto ins : group)
            {
                // std::cout << "dot_ins: " << std::endl;
                // m.debug_print(ins);
                if(std::distance(m.begin(), ins) < std::distance(m.begin(), insert_pos))
                    insert_pos = ins;
            }

            // 5b) Prepare unsqueezed inputs
            std::vector<instruction_ref> lhs_unsqueezed;
            std::vector<instruction_ref> rhs_unsqueezed;
            lhs_unsqueezed.reserve(group.size());
            rhs_unsqueezed.reserve(group.size());

            for(auto ins : group)
            {
                auto lhs = ins->inputs().at(0);
                auto rhs = ins->inputs().at(1);

                auto lhs_u = m.insert_instruction(
                    insert_pos, make_op("unsqueeze", {{"axes", {0}}}), lhs);
                auto rhs_u = m.insert_instruction(
                    insert_pos, make_op("unsqueeze", {{"axes", {0}}}), rhs);

                lhs_unsqueezed.push_back(lhs_u);
                rhs_unsqueezed.push_back(rhs_u);
            }

            // 5c) Concat along new batch axis (axis=0)
            auto lhs_cat = m.insert_instruction(
                insert_pos, make_op("concat", {{"axis", 0}}), lhs_unsqueezed);
            auto rhs_cat = m.insert_instruction(
                insert_pos, make_op("concat", {{"axis", 0}}), rhs_unsqueezed);

            // Optional: ensure contiguous layout if needed by dot
            lhs_cat = m.insert_instruction(insert_pos, make_op("contiguous"), lhs_cat);
            rhs_cat = m.insert_instruction(insert_pos, make_op("contiguous"), rhs_cat);

            // 5d) Single batched dot
            auto fused = m.insert_instruction(insert_pos, make_op("dot"), lhs_cat, rhs_cat);

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
                // auto out_i = m.insert_instruction(
                //     insert_pos, make_op("squeeze", {{"axes", {0}}}), slice_i);
                

                // Replace original dot with its fused output
                m.replace_instruction(group[i], out_i);
                // std::cout << "replaced ins: " << std::endl;
                // m.debug_print(out_i);
            }
            // std::cout << "new program" << std::endl;
            // m.debug_print();
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
